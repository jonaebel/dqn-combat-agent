package nn.rl;

import nn.core.NeuronalNetwork;
import nn.math.activationFunctions.LeakyReLu;
import nn.math.activationFunctions.Linear;
import nn.math.activationFunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

/*
 Converts a turn-based combat state into a normalised double[] vector
 suitable as input for the neural network.

 All values are scaled to [0, 1] so no single feature dominates
 gradient updates. The encoding includes:
   - Per-fighter: relative HP, relative stats, one-hot element type
   - Per-side:    alive-ratio
   - Active matchup: conditions, stat stages, type effectiveness, speed ratio,
                     estimated best-move damage, action uses remaining
*/
public class StateEncoder {

    private static final List<Double> vec = new ArrayList<>();

    private StateEncoder() {}

    public static double[] encode(Fighter[] team, Fighter[] enemyTeam,
                                  Fighter active, Fighter enemyActive,
                                  int teamSize) {
        vec.clear();

        Fighter[] all = new Fighter[team.length + enemyTeam.length];
        System.arraycopy(team,      0, all, 0,           team.length);
        System.arraycopy(enemyTeam, 0, all, team.length, enemyTeam.length);

        for (Fighter f : team)      encodeFighter(all, f);
        for (Fighter f : enemyTeam) encodeFighter(all, f);

        encodeAliveRatio(team, enemyTeam, teamSize);

        encodeCondition(active);
        encodeStatStages(active);
        encodeCondition(enemyActive);
        encodeStatStages(enemyActive);

        encodeTypeEffectiveness(active, enemyActive);
        encodeTypeEffectiveness(enemyActive, active);
        encodeSpeedRatio(active, enemyActive);

        encodeDamageEstimate(active, enemyActive);
        encodeDamageEstimate(enemyActive, active);

        encodeActionUses(active);

        double[] encoded = new double[vec.size()];
        for (int i = 0; i < encoded.length; i++) encoded[i] = vec.get(i);
        return encoded;
    }

    // ---------- per-fighter features ----------

    private static void encodeFighter(Fighter[] all, Fighter f) {
        encodeHp(f);
        encodeAttack(all, f);
        encodeDefense(all, f);
        encodeSpeed(all, f);
        encodeType(f);
    }

    private static void encodeHp(Fighter f) {
        vec.add(f.getHp() / f.getMaxHp());
    }

    private static void encodeAttack(Fighter[] all, Fighter f) {
        double max = 0;
        for (Fighter x : all) if (x.getAttack() > max) max = x.getAttack();
        vec.add(max > 0 ? f.getAttack() / max : 0.0);
    }

    private static void encodeDefense(Fighter[] all, Fighter f) {
        double max = 0;
        for (Fighter x : all) if (x.getDefense() > max) max = x.getDefense();
        vec.add(max > 0 ? f.getDefense() / max : 0.0);
    }

    private static void encodeSpeed(Fighter[] all, Fighter f) {
        int max = 0;
        for (Fighter x : all) if (x.getSpeed() > max) max = x.getSpeed();
        vec.add(max > 0 ? (double) f.getSpeed() / max : 0.0);
    }

    private static void encodeType(Fighter f) {
        for (ElementType t : ElementType.values()) {
            boolean has = false;
            for (ElementType ft : f.getType()) if (ft == t) { has = true; break; }
            vec.add(has ? 1.0 : 0.0);
        }
    }

    // ---------- team-level features ----------

    private static void encodeAliveRatio(Fighter[] team, Fighter[] enemyTeam, int teamSize) {
        int myAlive = 0, enemyAlive = 0;
        for (Fighter f : team)      if (f.isAlive()) myAlive++;
        for (Fighter f : enemyTeam) if (f.isAlive()) enemyAlive++;
        vec.add((double) myAlive    / teamSize);
        vec.add((double) enemyAlive / teamSize);
    }

    // ---------- active matchup features ----------

    private static void encodeCondition(Fighter f) {
        Condition current = f.getEffect();
        for (Condition c : Condition.values()) vec.add(current == c ? 1.0 : 0.0);
    }

    private static void encodeStatStages(Fighter f) {
        vec.add((f.getAttackStat()  + 2) / 4.0);
        vec.add((f.getDefenseStat() + 2) / 4.0);
        vec.add((f.getSpeedStat()   + 2) / 4.0);
    }

    private static void encodeTypeEffectiveness(Fighter attacker, Fighter defender) {
        double best = 0.0;
        for (Action a : attacker.getMoves()) {
            double eff = Combat.typeMultiplier(a, defender);
            if (eff > best) best = eff;
        }
        vec.add(best / 4.0); // max possible is 2×2 = 4.0
    }

    private static void encodeSpeedRatio(Fighter a, Fighter b) {
        double total = a.getSpeed() + b.getSpeed();
        vec.add(total > 0 ? a.getSpeed() / total : 0.5);
    }

    private static void encodeDamageEstimate(Fighter attacker, Fighter defender) {
        double effAtk = attacker.getAttack() * Combat.statMultiplier(attacker.getAttackStat());
        double effDef = defender.getDefense() * Combat.statMultiplier(defender.getDefenseStat());
        double best = 0.0;
        for (Action a : attacker.getMoves()) {
            if (a.getMoveDamage() <= 0) continue;
            double dmg = a.getMoveDamage() * (2.0 * effAtk / (effAtk + effDef))
                         * Combat.typeMultiplier(a, defender);
            if (dmg > best) best = dmg;
        }
        vec.add(Math.min(best / Math.max(defender.getMaxHp(), 1), 1.0));
    }

    private static void encodeActionUses(Fighter f) {
        Action[] actions = f.getMoves();
        for (int i = 0; i < 4; i++) {
            if (i < actions.length && actions[i] != null)
                vec.add(Math.min(actions[i].getUses() / 5.0, 1.0));
            else
                vec.add(0.0);
        }
    }

    // ---------- entry point / demo ----------

    public static void main(String[] args) {
        /*
         Input size for team_size=3:
           Per fighter: HP(1) + atk(1) + def(1) + spd(1) + type(8) = 12
           Both teams:  12 * 2 * 3 = 72
           Alive ratio: 2
           Conditions:  6 * 2 = 12
           Stat stages: 3 * 2 = 6
           Type eff:    2
           Speed ratio: 1
           Dmg est:     2
           Action uses: 4
           Total:       101
        */
        int INPUT_SIZE = 101;
        int TEAM_SIZE  = 3;
        int[] structure = {256, 128, 6};
        ActivationFunction[] activations = {new LeakyReLu(), new LeakyReLu(), new Linear()};

        NeuronalNetwork agent_a  = new NeuronalNetwork(structure, INPUT_SIZE, activations);
        NeuronalNetwork agent_b  = new NeuronalNetwork(structure, INPUT_SIZE, activations);
        NeuronalNetwork target_a = new NeuronalNetwork(structure, INPUT_SIZE, activations);
        NeuronalNetwork target_b = new NeuronalNetwork(structure, INPUT_SIZE, activations);

        Gym gym = new Gym(agent_a, agent_b, target_a, target_b, TEAM_SIZE, 1000, structure, INPUT_SIZE, activations);

        System.out.println("Phase 1: Agent A learns vs random opponent");
        gym.run(5000, 32, 0.001, 0.99, 10, false);

        System.out.println("Phase 2: Self-play — both agents learn together");
        gym.run(10000, 32, 0.001, 0.99, 10, true);
    }
}
