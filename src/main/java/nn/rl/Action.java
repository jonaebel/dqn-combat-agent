package nn.rl;

public class Action {

    private final String name;
    private final int damage;
    private final ElementType type;
    private int uses;
    private final int maxUses;

    public Action(String name, int damage, ElementType type, int maxUses) {
        this.name = name;
        this.damage = damage;
        this.type = type;
        this.maxUses = maxUses;
        this.uses = maxUses;
    }

    public void useAction() {
        if (uses > 0) uses--;
    }

    public void reset() {
        uses = maxUses;
    }

    public String getName()     { return name; }
    public int getMoveDamage()  { return damage; }
    public ElementType getType(){ return type; }
    public int getUses()        { return uses; }
    public int getMaxUses()     { return maxUses; }
}
