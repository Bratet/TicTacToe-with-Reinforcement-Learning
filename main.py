from models import MDP, MonteCarlo, TemporalDifference

if __name__ == "__main__":
    print("Choose an algorithm:")
    print("1. MDP")
    print("2. Monte Carlo")
    print("3. Temporal Difference")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        game = MDP()
    elif choice == 2:
        game = MonteCarlo()
    elif choice == 3:
        game = TemporalDifference()
    else:
        print("Invalid choice. Exiting...")
        exit()

    game.play()
