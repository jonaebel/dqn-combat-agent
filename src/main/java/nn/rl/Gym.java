package nn.rl;

import nn.core.NeuronalNetwork;
import nn.core.ReplayBuffer;
import nn.math.activationFunctions.ActivationFunction;
import nn.math.lossFunctions.Huber;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/*
 DQN training loop for a turn-based 1v1 combat environment.

 Two agents (A and B) each control a team of fighters.
 Each turn the active agent picks one of up to 6 actions:
   0-3  use action slot 0-3
   4-5  switch to bench fighter 0 or 1

 Training modes:
   self_play = false  — only agent A learns; opponents come from the pool
   self_play = true   — both agents learn; pool opponents are mixed in for diversity

 Key DQN components used:
   - epsilon-greedy exploration with exponential decay
   - separate target network (synced every `sync_every` episodes)  ← DQN concept
   - opponent pool of historical snapshots (AlphaGo-style)          ← separate concept
   - experience replay via ReplayBuffer
   - Huber loss for stable Q-value regression
*/
public class Gym {

    private final NeuronalNetwork agent_a;
    private final NeuronalNetwork agent_b;
    private NeuronalNetwork target_a;
    private NeuronalNetwork target_b;

    private static final Random RANDOM = new Random();
    private final CombatEnvironment env;
    private final ReplayBuffer buffer_a;
    private final ReplayBuffer buffer_b;
    private final int TEAM_SIZE;

    // Network architecture — needed to clone snapshots into the pool
    private final int[] structure;
    private final int inputSize;
    private final ActivationFunction[] activations;

    // Opponent pool: historical snapshots of past agents (AlphaGo-style)
    private final List<NeuronalNetwork> opponentPool = new ArrayList<>();
    private static final int MAX_POOL_SIZE = 20;

    public Gym(NeuronalNetwork agent_a, NeuronalNetwork agent_b,
               NeuronalNetwork target_a, NeuronalNetwork target_b,
               int team_size, int buffer_capacity,
               int[] structure, int inputSize, ActivationFunction[] activations) {
        this.agent_a = agent_a;
        this.agent_b = agent_b;
        this.target_a = target_a;
        this.target_b = target_b;
        this.target_a.copy_weights_from(agent_a);
        this.target_b.copy_weights_from(agent_b);
        this.TEAM_SIZE = team_size;
        this.env = new CombatEnvironment(team_size);
        this.buffer_a = new ReplayBuffer(buffer_capacity);
        this.buffer_b = new ReplayBuffer(buffer_capacity);
        this.structure   = structure;
        this.inputSize   = inputSize;
        this.activations = activations;
    }

    /** Erstellt einen eingefrorenen Snapshot eines Agenten für den Pool. */
    private NeuronalNetwork makeSnapshot(NeuronalNetwork source) {
        NeuronalNetwork snap = new NeuronalNetwork(structure, inputSize, activations);
        snap.copy_weights_from(source);
        return snap;
    }

    public static boolean game_over(Fighter[] team) {
        for (Fighter f : team) if (f.isAlive()) return false;
        return true;
    }

    public static int choose_action(NeuronalNetwork agent, Fighter[] team, Fighter[] enemyTeam,
                                    Fighter active, Fighter enemyActive,
                                    int teamSize, double epsilon) {
        if (RANDOM.nextDouble() < epsilon) return RANDOM.nextInt(6);

        double[] input  = StateEncoder.encode(team, enemyTeam, active, enemyActive, teamSize);
        double[] output = agent.feed_forward(input);
        int best = 0;
        for (int i = 1; i < output.length; i++)
            if (output[i] > output[best]) best = i;
        return best;
    }

    public static Fighter execute_action(Fighter active, Fighter enemyActive,
                                         int index, List<Fighter> switchOptions) {
        switch (index) {
            case 0 -> Combat.attack(active, active.getMoves()[0], enemyActive);
            case 1 -> Combat.attack(active, active.getMoves()[1], enemyActive);
            case 2 -> Combat.attack(active, active.getMoves()[2], enemyActive);
            case 3 -> Combat.attack(active, active.getMoves()[3], enemyActive);
            case 4 -> { return switchOptions.isEmpty()   ? active : switchOptions.get(0); }
            case 5 -> { return switchOptions.size() < 2  ? (switchOptions.isEmpty() ? active : switchOptions.get(0))
                                                          : switchOptions.get(1); }
        }
        return active;
    }

    public Fighter forced_switch(NeuronalNetwork agent, Fighter[] team, Fighter[] enemyTeam,
                                  Fighter active, Fighter enemyActive) {
        List<Fighter> options = switch_options(Arrays.asList(team), active);
        if (options.isEmpty())  return active;
        if (options.size() == 1) return options.get(0);

        double[] input  = StateEncoder.encode(team, enemyTeam, active, enemyActive, team.length);
        double[] output = agent.feed_forward(input);
        return output[4] >= output[5] ? options.get(0) : options.get(1);
    }

    public static List<Fighter> switch_options(List<Fighter> team, Fighter active) {
        List<Fighter> options = new ArrayList<>();
        for (Fighter f : team) if (f != active && f.isAlive()) options.add(f);
        return options;
    }

    public static double total_hp_ratio(Fighter[] team) {
        double hp = 0, max = 0;
        for (Fighter f : team) { max += f.getMaxHp(); hp += f.getHp(); }
        return hp / max;
    }

    public static int count_alive(Fighter[] team) {
        int count = 0;
        for (Fighter f : team) if (f.isAlive()) count++;
        return count;
    }

    public static double calculate_reward(Fighter[] team, Fighter[] enemyTeam,
                                           CombatEnvironment.Snapshot snap,
                                           boolean won, boolean lost) {
        double reward = 0.0;
        reward += (snap.hpRatioB() - total_hp_ratio(enemyTeam)) * 2.0;
        reward -= (snap.hpRatioA() - total_hp_ratio(team))      * 2.0;
        reward += snap.aliveB() - count_alive(enemyTeam);
        reward -= snap.aliveA() - count_alive(team);
        if (won)  reward += 5;
        if (lost) reward -= 5;
        return reward;
    }

    public double run_episode(NeuronalNetwork agentA, NeuronalNetwork agentB,
                               int teamSize, double epsilonA, double epsilonB) {
        env.reset(teamSize);
        final int MAX_TURNS = 200;
        int turn = 0;

        while (!game_over(env.teamA) && !game_over(env.teamB) && turn++ < MAX_TURNS) {
            CombatEnvironment.Snapshot snap = env.snapshot();

            double[] stateA = null, stateB = null;
            int actionA = -1, actionB = -1;

            boolean aFirst = env.activeA.getSpeed() > env.activeB.getSpeed() ||
                             (env.activeA.getSpeed() == env.activeB.getSpeed() && RANDOM.nextBoolean());

            if (aFirst) {
                stateA  = StateEncoder.encode(env.teamA, env.teamB, env.activeA, env.activeB, teamSize);
                actionA = choose_action(agentA, env.teamA, env.teamB, env.activeA, env.activeB, teamSize, epsilonA);
                env.activeA = execute_action(env.activeA, env.activeB, actionA,
                              switch_options(Arrays.asList(env.teamA), env.activeA));

                if (env.activeB.isAlive()) {
                    stateB  = StateEncoder.encode(env.teamB, env.teamA, env.activeB, env.activeA, teamSize);
                    actionB = choose_action(agentB, env.teamB, env.teamA, env.activeB, env.activeA, teamSize, epsilonB);
                    env.activeB = execute_action(env.activeB, env.activeA, actionB,
                                  switch_options(Arrays.asList(env.teamB), env.activeB));
                } else {
                    env.activeB = forced_switch(agentB, env.teamB, env.teamA, env.activeB, env.activeA);
                }
                if (!env.activeA.isAlive())
                    env.activeA = forced_switch(agentA, env.teamA, env.teamB, env.activeA, env.activeB);
            } else {
                stateB  = StateEncoder.encode(env.teamB, env.teamA, env.activeB, env.activeA, teamSize);
                actionB = choose_action(agentB, env.teamB, env.teamA, env.activeB, env.activeA, teamSize, epsilonB);
                env.activeB = execute_action(env.activeB, env.activeA, actionB,
                              switch_options(Arrays.asList(env.teamB), env.activeB));

                if (env.activeA.isAlive()) {
                    stateA  = StateEncoder.encode(env.teamA, env.teamB, env.activeA, env.activeB, teamSize);
                    actionA = choose_action(agentA, env.teamA, env.teamB, env.activeA, env.activeB, teamSize, epsilonA);
                    env.activeA = execute_action(env.activeA, env.activeB, actionA,
                                  switch_options(Arrays.asList(env.teamA), env.activeA));
                } else {
                    env.activeA = forced_switch(agentA, env.teamA, env.teamB, env.activeA, env.activeB);
                }
                if (!env.activeB.isAlive())
                    env.activeB = forced_switch(agentB, env.teamB, env.teamA, env.activeB, env.activeA);
            }

            double[] nextStateA = StateEncoder.encode(env.teamA, env.teamB, env.activeA, env.activeB, teamSize);
            double[] nextStateB = StateEncoder.encode(env.teamB, env.teamA, env.activeB, env.activeA, teamSize);
            boolean done = game_over(env.teamA) || game_over(env.teamB);

            double rewardA = calculate_reward(env.teamA, env.teamB, snap,
                                               game_over(env.teamB), game_over(env.teamA));
            CombatEnvironment.Snapshot snapB = new CombatEnvironment.Snapshot(
                    snap.hpRatioB(), snap.hpRatioA(), snap.aliveB(), snap.aliveA());
            double rewardB = calculate_reward(env.teamB, env.teamA, snapB,
                                               game_over(env.teamA), game_over(env.teamB));

            if (actionA != -1) { buffer_a.add(stateA, actionA, rewardA, nextStateA, done); }
            if (actionB != -1) { buffer_b.add(stateB, actionB, rewardB, nextStateB, done); }
        }

        return game_over(env.teamB) ? 1.0 : 0.0;
    }

    public void train_step(NeuronalNetwork agent, NeuronalNetwork target,
                            ReplayBuffer buffer, int batch_size, double lr, double gamma) {
        if (buffer.get_size() < batch_size) return;
        Huber huber = new Huber();
        for (ReplayBuffer.Transition t : buffer.sample(batch_size)) {
            double[] qValues  = agent.feed_forward(t.state());
            double[] nextQ    = target.feed_forward(t.next_state());
            double maxNextQ   = Double.NEGATIVE_INFINITY;
            for (double q : nextQ) maxNextQ = Math.max(maxNextQ, q);
            double targetQ    = t.done() ? t.reward() : t.reward() + gamma * maxNextQ;
            double[] targets  = qValues.clone();
            targets[t.action()] = targetQ;
            agent.train(t.state(), targets, lr, huber);
        }
    }

    public void run(int episodes, int batch_size, double lr, double gamma,
                    int sync_every, boolean self_play) {
        double reward_sum = 0;
        double epsilon = 1.0;
        final double EPSILON_MIN   = 0.05;
        final double EPSILON_DECAY = 0.995;

        for (int episode = 0; episode < episodes; episode++) {
            epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);

            // --- Gegnerauswahl ---
            // Pool-Mitglieder sind eingefroren → spielen greedy (epsilon = 0 -> keine varianz oder neues ausprobieren).
            // Ist der Pool noch leer, fällt es auf agent_b zurück (random oder lernend).
            NeuronalNetwork opponent;
            double epsilonB;
            if (!opponentPool.isEmpty()) {
                opponent = opponentPool.get(RANDOM.nextInt(opponentPool.size()));
                epsilonB = 0.0;
            } else {
                opponent = agent_b;
                epsilonB = self_play ? epsilon : 1.0;
            }
            boolean vsPool = opponent != agent_b;

            // --- Episode ---
            reward_sum += run_episode(agent_a, opponent, TEAM_SIZE, epsilon, epsilonB);

            // --- Training ---
            train_step(agent_a, target_a, buffer_a, batch_size, lr, gamma);
            // agent_b nur trainieren wenn er wirklich gespielt hat (kein eingefrorenes Pool-Mitglied)
            if (self_play && !vsPool)
                train_step(agent_b, target_b, buffer_b, batch_size, lr, gamma);

            // --- Target-Netze synchronisieren (DQN-Konzept, unabhängig vom Pool) ---
            if (episode % sync_every == 0) {
                target_a.copy_weights_from(agent_a);
                if (self_play) target_b.copy_weights_from(agent_b);
            }

            // --- Pool befüllen + Logging alle 100 Episoden ---
            if (episode % 100 == 0 && episode > 0) {
                opponentPool.add(makeSnapshot(agent_a));
                if (self_play) opponentPool.add(makeSnapshot(agent_b));
                while (opponentPool.size() > MAX_POOL_SIZE) opponentPool.remove(0);

                String mode = self_play ? "Self-Play+Pool" : "vs Pool";
                System.out.printf("Episode %d | %s | Win-Rate: %.1f%% | Epsilon: %.3f | Pool: %d%n",
                        episode, mode, reward_sum, epsilon, opponentPool.size());
                reward_sum = 0;
            }
        }
    }

    // -----------------------------------------------------------------------

    static class CombatEnvironment {

        private final Fighter[] POOL;
        Fighter[] teamA;
        Fighter[] teamB;
        Fighter activeA;
        Fighter activeB;

        record Snapshot(double hpRatioA, double hpRatioB, int aliveA, int aliveB) {}

        CombatEnvironment(int teamSize) {
            FighterClass[] all = FighterClass.values();
            POOL = new Fighter[all.length];
            for (int i = 0; i < all.length; i++) POOL[i] = all[i].create();
            teamA = new Fighter[teamSize];
            teamB = new Fighter[teamSize];
        }

        void reset(int teamSize) {
            for (Fighter f : POOL) f.reset();
            randomize(teamSize);
        }

        private void randomize(int teamSize) {
            Integer[] idx = new Integer[POOL.length];
            for (int i = 0; i < POOL.length; i++) idx[i] = i;

            Collections.shuffle(Arrays.asList(idx));
            for (int i = 0; i < teamSize; i++) teamA[i] = POOL[idx[i]].copy();

            Collections.shuffle(Arrays.asList(idx));
            for (int i = 0; i < teamSize; i++) teamB[i] = POOL[idx[i]].copy();

            activeA = teamA[0];
            activeB = teamB[0];
        }

        Snapshot snapshot() {
            return new Snapshot(
                    Gym.total_hp_ratio(teamA), Gym.total_hp_ratio(teamB),
                    Gym.count_alive(teamA),    Gym.count_alive(teamB));
        }
    }
}
