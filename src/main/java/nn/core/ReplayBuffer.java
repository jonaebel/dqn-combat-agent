package nn.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/*
 Circular experience replay buffer for DQN training.
 Stores the last N (state, action, reward, next_state, done) transitions.
 Random sampling breaks temporal correlations and stabilises training.
*/
public class ReplayBuffer {
    public record Transition(double[] state, int action, double reward, double[] next_state, boolean done){}
    private int head;
    private int size;
    private Transition[] buffer;
    private final int CAPACITY;
    private final Random RANDOM = new Random();

    private List<Transition> transition_list = new ArrayList<>();

    public ReplayBuffer(int capacity) {
        head = 0;
        size = 0;
        buffer = new Transition[capacity];
        this.CAPACITY = capacity;

    }
    public void add(double[] state, int action, double reward, double[] next_state, boolean done) {
        buffer[head] = new Transition(state, action, reward, next_state, done);
        head = (head+1) % CAPACITY;
        size = Math.min(size+1, CAPACITY);
    }

    public Transition[] sample(int batch_size) {
        transition_list.clear();
        for(int i = 0; i < size; i++) transition_list.add(buffer[i]);
        Collections.shuffle(transition_list, RANDOM);
        Transition[] sample_transitions = new Transition[batch_size];
        for(int i = 0; i < batch_size; i++) {
            sample_transitions[i] = transition_list.get(i);
        }
        return sample_transitions;
    }

    public int get_size() {
        return this.size;
    }
}


/*
Replay Buffer saves the Last N Game histories -> Modulo
*/