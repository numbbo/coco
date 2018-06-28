import sys


def read_and_write(obj=1):
    """Test function that reads the values from the file 'variables.txt', sums them and outputs the
    result the the file 'objectives.txt'.
    """
    if obj > 2:
        raise ValueError("The number of objectives ({}) cannot be larger than 2".format(obj))

    # Read the variables
    with open('variables.txt') as f:
        content = f.readlines()
        content = [1 + float(line.rstrip('\n')) for line in content]

    # Compute the result
    variable_sum = abs(sum(content[1:]))

    # Write the result
    with open('objectives.txt', 'w') as f:
        f.write('{}\n'.format(obj))
        f.write('{}\n'.format(variable_sum))
        if obj == 2:
            f.write('{}\n'.format(10 - variable_sum))


if __name__ == '__main__':
    read_and_write(int(sys.argv[1]))
    #for i in range(10000):
    #    print(i)
    #    read_and_write(int(sys.argv[1]))
