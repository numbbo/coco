import re 
import pickle


def read_seed_file(filename):
    with open(filename, "r") as file_:
        file_string = file_.read()
    file_string = [line for line in file_string.split("\n") if line ]
    file_string = [line.split(',') for line in file_string]
    file_string = [line for line in file_string if len(line) == 2]
    seeds_dictionary = {function[0]: int(re.findall(".*: (\d*)", function[1])[0]) for function in file_string}
    return seeds_dictionary

def test_seeds(seeds, legacy_seeds):
    assert (len(seeds) == len(legacy_seeds) and len(legacy_seeds) > 0) 
    failed_test_counter = 0
    passed_test_counter = 0 
    for fun_id, seed in seeds.items():
        legacy_seed = legacy_seeds[fun_id]
        try:
            assert (legacy_seed == seed), f"{fun_id} failed test... seed: {seed}, legacy seed: {int(legacy_seed)}"
            passed_test_counter += 1
        except AssertionError as error:
            print(error)
            failed_test_counter += 1
    return failed_test_counter, passed_test_counter

if __name__ == "__main__":
    seed_path_legacy = "code-experiments/test/regression-test/regression-test-bbob-noisy/data_legacy/bbob_noisy_seeds.json"
    seed_path = "code-experiments/test/regression-test/regression-test-bbob-noisy/data/bbob_noisy_seeds.txt"
    with open(seed_path_legacy, "rb") as file_:
        seeds_legacy = pickle.load(file_, encoding='latin1')
    seeds = read_seed_file(seed_path)
    failed_test_counter, passed_test_counter = test_seeds(seeds, seeds_legacy)
    print(f"Execution terminated with {failed_test_counter} failed tests and {passed_test_counter} passed tests")

