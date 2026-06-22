package nn.rl;

/*
 Example fighter roster for the RL training environment.
 Each entry creates a Fighter with distinct stats and action pool
 to give the agent meaningful strategic choices (type matchups,
 speed tiers, bulk vs. power trade-offs).
*/
public enum FighterClass {

    STRIKER {
        @Override
        public Fighter create() {
            return new ConcreteFighter(
                    200, 90, 50, 80,
                    new ElementType[]{ElementType.FIRE},
                    new Action[]{
                            new Action("Flame Strike",  70, ElementType.FIRE,  5),
                            new Action("Ember",         40, ElementType.FIRE,  8),
                            new Action("Heat Wave",     55, ElementType.FIRE,  6),
                            new Action("Quick Slash",   35, ElementType.VOID,  10)
                    }
            );
        }
    },

    TANK {
        @Override
        public Fighter create() {
            return new ConcreteFighter(
                    320, 55, 110, 35,
                    new ElementType[]{ElementType.EARTH},
                    new Action[]{
                            new Action("Rock Crush",    60, ElementType.EARTH, 6),
                            new Action("Ground Slam",   80, ElementType.EARTH, 4),
                            new Action("Dust Wave",     45, ElementType.EARTH, 8),
                            new Action("Body Block",    30, ElementType.VOID,  10)
                    }
            );
        }
    },

    SPEEDSTER {
        @Override
        public Fighter create() {
            return new ConcreteFighter(
                    180, 75, 60, 120,
                    new ElementType[]{ElementType.WIND},
                    new Action[]{
                            new Action("Gust Blade",    50, ElementType.WIND,  7),
                            new Action("Cyclone",       75, ElementType.WIND,  4),
                            new Action("Tail Wind",     35, ElementType.WIND,  10),
                            new Action("Swift Strike",  45, ElementType.VOID,  8)
                    }
            );
        }
    },

    MAGE {
        @Override
        public Fighter create() {
            return new ConcreteFighter(
                    160, 105, 45, 65,
                    new ElementType[]{ElementType.PSYCHIC},
                    new Action[]{
                            new Action("Mind Blast",    85, ElementType.PSYCHIC, 4),
                            new Action("Psi Wave",      55, ElementType.PSYCHIC, 7),
                            new Action("Void Pulse",    65, ElementType.VOID,    5),
                            new Action("Focus Ray",     40, ElementType.PSYCHIC, 9)
                    }
            );
        }
    },

    BALANCED {
        @Override
        public Fighter create() {
            return new ConcreteFighter(
                    240, 70, 70, 70,
                    new ElementType[]{ElementType.WATER},
                    new Action[]{
                            new Action("Aqua Jet",      55, ElementType.WATER, 7),
                            new Action("Tidal Surge",   75, ElementType.WATER, 4),
                            new Action("Mist Veil",     40, ElementType.WATER, 9),
                            new Action("Splash",        30, ElementType.VOID,  10)
                    }
            );
        }
    };

    public abstract Fighter create();

    private static class ConcreteFighter extends Fighter {
        public ConcreteFighter(double maxHp, double attack, double defense, int speed,
                               ElementType[] types, Action[] actions) {
            super(maxHp, attack, defense, speed, types, actions);
        }

        @Override
        public Fighter copy() {
            Action[] copiedActions = new Action[actions.length];
            for (int i = 0; i < actions.length; i++) {
                Action a = actions[i];
                copiedActions[i] = new Action(a.getName(), a.getMoveDamage(), a.getType(), a.getMaxUses());
            }
            return new ConcreteFighter(maxHp, attack, defense, speed, types, copiedActions);
        }
    }
}
