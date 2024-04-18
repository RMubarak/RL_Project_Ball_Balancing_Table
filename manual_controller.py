
def manual_policy(state):
    """Allows the user to input commands through the Terminal or a Python Console, the interface provides the current state and returns the command as an int to the game. This interface can handle the w, a, s, d, and x keys as commands to move the table servomotors in the x and y directions. 
    To run this policy... TODO
    
    possible commands:
            -x = 4  OR 'a'
            +x = 3 OR 'd'
            +y = 1 OR 'w'    
            -y = 2 OR 'x'
            Nothing  = 0 or 's'

    Args:
        state: The current state to be displayed to the user

    Returns:
        int: The command that will be used in the game
    """
    
    # Shows the current state and asks the user to provide a command
    print("The current state is", state)
    print("Please input the next command:")
    
    command = input()
    
    # To ensure you can provide commands using letters as well as ints
    if isinstance(command,str):
        
        if command == 'a':
            command = 4
        if command == 'd':
            command = 3
        if command == 'w':
            command = 1
        if command == 'x':
            command = 2
        if command == 's':
            command = 0
        
    try:
        command = int(command)
            
    except ValueError:
        print("Invalid command, please try again!")
        
    return command