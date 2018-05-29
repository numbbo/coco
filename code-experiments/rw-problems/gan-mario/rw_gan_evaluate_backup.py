if __name__ == '__main__':
    path = ''

    # Read the variables
    with open(path + 'variables.txt') as f:
        content = f.readlines()
        content = [float(line.rstrip('\n')) for line in content]
        f.close()

    # Compute the result
    variable_sum = abs(sum(content[1:]))

    # Write the result
    with open(path + 'objectives.txt', 'w') as f:
        f.write('{}\n'.format(1))
        f.write('{}\n'.format(variable_sum))
        f.close()

