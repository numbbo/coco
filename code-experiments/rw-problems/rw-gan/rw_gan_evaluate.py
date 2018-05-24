if __name__ == '__main__':
    expected_num_variables = 32
    path = ''

    # Read the variables
    with open(path + 'variables.txt') as f:
        content = f.readlines()
        content = [float(line.rstrip('\n')) for line in content]
        num_variables = content[0]
        if num_variables != expected_num_variables:
            raise ValueError("num_variables should be '{}', but is '{}'"
                             "".format(expected_num_variables, num_variables))
        f.close()

    # Compute the result
    variable_sum = sum(content[1:])

    # Write the result
    with open(path + 'objectives.txt', 'w') as f:
        f.write('{}\n'.format(1))
        f.write('{}\n'.format(variable_sum))
        f.close()
