package nn.rl;

/*
 Minimal combat resolver for the RL environment.
 Handles damage calculation with type multipliers and stat stages.
*/
public class Combat {

    /*
     Simple 2x2 type advantage chart.
     Super-effective = 2.0, not-very-effective = 0.5, neutral = 1.0.
     Multiplicative when a defender has two types.
    */
    public static double typeMultiplier(Action action, Fighter defender) {
        double multiplier = 1.0;
        for (ElementType defType : defender.getType()) {
            multiplier *= singleMatchup(action.getType(), defType);
        }
        return multiplier;
    }

    private static double singleMatchup(ElementType atk, ElementType def) {
        return switch (atk) {
            case FIRE    -> def == ElementType.EARTH  ? 2.0 : def == ElementType.WATER  ? 0.5 : 1.0;
            case WATER   -> def == ElementType.FIRE   ? 2.0 : def == ElementType.EARTH  ? 0.5 : 1.0;
            case EARTH   -> def == ElementType.WIND   ? 2.0 : def == ElementType.FIRE   ? 0.5 : 1.0;
            case WIND    -> def == ElementType.WATER  ? 2.0 : def == ElementType.EARTH  ? 0.5 : 1.0;
            case LIGHT   -> def == ElementType.DARK   ? 2.0 : def == ElementType.VOID   ? 0.5 : 1.0;
            case DARK    -> def == ElementType.PSYCHIC? 2.0 : def == ElementType.LIGHT  ? 0.5 : 1.0;
            case PSYCHIC -> def == ElementType.VOID   ? 2.0 : def == ElementType.DARK   ? 0.5 : 1.0;
            case VOID    -> def == ElementType.LIGHT  ? 2.0 : def == ElementType.PSYCHIC? 0.5 : 1.0;
        };
    }

    public static void attack(Fighter attacker, Action action, Fighter defender) {
        double effAtk = attacker.getAttack() * statMultiplier(attacker.getAttackStat());
        double effDef = defender.getDefense() * statMultiplier(defender.getDefenseStat());
        double multi  = typeMultiplier(action, defender);
        double dmg    = action.getMoveDamage() * (2.0 * effAtk / (effAtk + effDef)) * multi;
        defender.setHp(defender.getHp() - dmg);
        action.useAction();
    }

    public static double statMultiplier(int stage) {
        return switch (stage) {
            case  2 -> 1.5;
            case  1 -> 1.25;
            case -1 -> 0.75;
            case -2 -> 0.5;
            default -> 1.0;
        };
    }
}
