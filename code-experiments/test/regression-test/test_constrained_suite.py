from test_suites import *

if __name__ == "__main__":
    try:
        ndata = int(sys.argv[1])
    except IndexError:
        ndata = 2
    except ValueError:
        print(__doc__)
    try:
        ndata
    except:
        pass
    else:
        name = "bbob-constrained"
        cocoex.known_suite_names.append(name)
        data_file_path = ("data/regression_test_%ddata_for_suite_" % ndata) + name + ".py"

        if not os.path.exists(data_file_path):
            remote_data_path = 'http://coco.gforge.inria.fr/regression-tests/'
            # download data from remote_data_path:
            if not os.path.exists(os.path.split(data_file_path)[0]):
                try:
                    os.makedirs(os.path.split(data_file_path)[0])
                except os.error:  # python 2&3 compatible
                    raise
            url = '/'.join((remote_data_path, data_file_path))
            print("  downloading %s to %s" % (url, data_file_path))
            urlretrieve(url, data_file_path)

        regression_test_a_suite(name, data_file_path)