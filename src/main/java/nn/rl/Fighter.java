package nn.rl;

public abstract class Fighter {

    protected double hp;
    protected final double maxHp;
    protected final double attack;
    protected final double defense;
    protected final int speed;
    protected final ElementType[] types;
    protected final Action[] actions;

    protected Condition condition = Condition.NONE;
    protected int attackStat  = 0;  // -2 to +2
    protected int defenseStat = 0;
    protected int speedStat   = 0;

    protected Fighter(double maxHp, double attack, double defense, int speed,
                      ElementType[] types, Action[] actions) {
        this.maxHp   = maxHp;
        this.hp      = maxHp;
        this.attack  = attack;
        this.defense = defense;
        this.speed   = speed;
        this.types   = types;
        this.actions = actions;
    }

    public boolean isAlive()       { return hp > 0; }

    public void reset() {
        hp = maxHp;
        condition = Condition.NONE;
        attackStat = defenseStat = speedStat = 0;
        for (Action a : actions) a.reset();
    }

    public abstract Fighter copy();

    public double getHp()          { return hp; }
    public double getMaxHp()       { return maxHp; }
    public double getAttack()      { return attack; }
    public double getDefense()     { return defense; }
    public int getSpeed()          { return speed; }
    public ElementType[] getType() { return types; }
    public Action[] getMoves()     { return actions; }
    public Condition getEffect()   { return condition; }
    public int getAttackStat()     { return attackStat; }
    public int getDefenseStat()    { return defenseStat; }
    public int getSpeedStat()      { return speedStat; }
    public void setHp(double hp)   { this.hp = Math.max(0, hp); }
}
