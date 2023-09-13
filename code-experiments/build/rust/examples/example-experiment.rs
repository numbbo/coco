use coco_rs::{observer, suite, LogLevel, Observer, Problem, RandomState, Suite};

const BUDGET_MULTIPLIER: usize = 10;
const INDEPENDENT_RESTARTS: u64 = 1e5 as u64;
const RANDOM_SEED: u32 = 0xdeadbeef;

fn main() {
    let random_generator = &mut RandomState::new(RANDOM_SEED);
    println!("Running the example experiment... (might take time, be patient)");

    LogLevel::Info.set();

    example_experiment(
        suite::Name::Bbob,
        "",
        observer::Name::Bbob,
        "result_folder: RS_on_bbob",
        random_generator,
    );

    println!("Done!");
}

fn example_experiment(
    suite_name: suite::Name,
    suite_options: &str,
    observer_name: observer::Name,
    observer_options: &str,
    random_generator: &mut RandomState,
) {
    let suite = &mut Suite::new(suite_name, "", suite_options).unwrap();
    let observer = &mut Observer::new(observer_name, observer_options).unwrap();

    while let Some(problem) = &mut suite.next_problem(Some(observer)) {
        let dimension = problem.dimension();

        for _ in 1..=INDEPENDENT_RESTARTS {
            let evaluations_done = problem.evaluations() + problem.evaluations_constraints();
            let evaluations_remaining =
                (dimension * BUDGET_MULTIPLIER).saturating_sub(evaluations_done as usize);

            if problem.final_target_hit() || evaluations_remaining == 0 {
                break;
            }

            my_random_search(problem, evaluations_remaining, random_generator);
        }
    }
}

fn my_random_search(problem: &mut Problem, max_budget: usize, random_generator: &mut RandomState) {
    let dimension = problem.dimension();
    let number_of_objectives = problem.number_of_objectives();
    let numver_of_constraints = problem.number_of_constraints();
    let number_of_integer_variables = problem.number_of_integer_variables();
    let bounds = problem.get_ranges_of_interest();

    let x = &mut vec![0.0; dimension];
    let y = &mut vec![0.0; number_of_objectives];
    let c = &mut vec![0.0; numver_of_constraints];

    problem.initial_solution(x);
    problem.evaluate_function(x, y);

    for _ in 0..max_budget {
        for (i, xi) in x.iter_mut().enumerate() {
            let (lower, upper) = bounds[i].clone().into_inner();
            *xi = lower + random_generator.uniform() * (upper - lower);

            if i < number_of_integer_variables {
                *xi = xi.round();
            }
        }

        if numver_of_constraints > 0 {
            problem.evaluate_constraint(x, c);
        }

        problem.evaluate_function(x, y);
    }
}
