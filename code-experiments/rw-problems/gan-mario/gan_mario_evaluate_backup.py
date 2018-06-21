import sys

if __name__ == '__main__':
    obj = int(sys.argv[1])

    # Read the variables
    with open('variables.txt') as f:
        content = f.readlines()
        content = [1 + float(line.rstrip('\n')) for line in content]
        f.close()

    # Compute the result
    variable_sum = abs(sum(content[1:]))

    # Write the result
    with open('objectives.txt', 'w') as f:
        f.write('{}\n'.format(obj))
        f.write('{}\n'.format(variable_sum))
        if obj == 2:
            f.write('{}\n'.format(10 - variable_sum))
        f.close()

