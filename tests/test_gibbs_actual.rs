// Save this to tests/test_gibbs_actual.rs and run with: cargo test --test test_gibbs_actual -- --nocapture
use prak::common::rng::SimpleRng;
use prak::multisensor_lmbm::gibbs::multisensor_lmbm_gibbs_sampling;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    step3_gibbs: Step3Gibbs,
}

#[derive(Debug, Deserialize)]
struct Step3Gibbs {
    input: GibbsInput,
}

#[derive(Debug, Deserialize)]
struct GibbsInput {
    #[serde(rename = "L")]
    l: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "numberOfSamples")]
    number_of_samples: usize,
    rng_seed: u64,
}

fn flatten_l(l: &Vec<Vec<Vec<f64>>>) -> Vec<f64> {
    let dim1 = l.len();
    let dim2 = l[0].len();
    let dim3 = l[0][0].len();
    let mut flat = Vec::with_capacity(dim1 * dim2 * dim3);
    for k in 0..dim3 {
        for j in 0..dim2 {
            for i in 0..dim1 {
                flat.push(l[i][j][k]);
            }
        }
    }
    flat
}

#[test]
fn test_gibbs_directly_from_fixture() {
    let fixture_path = "tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json";
    let contents = std::fs::read_to_string(fixture_path).expect("Failed to read fixture");
    let fixture: Fixture = serde_json::from_str(&contents).expect("Failed to parse JSON");

    let input = &fixture.step3_gibbs.input;
    let l_flat = flatten_l(&input.l);
    let dimensions = vec![3, 12, 4];

    let mut rng = SimpleRng::new(input.rng_seed);
    let samples = multisensor_lmbm_gibbs_sampling(&mut rng, &l_flat, &dimensions, input.number_of_samples);

    println!("\nUsing ACTUAL library code:");
    println!("Number of unique samples: {}", samples.nrows());

    // Print first 5 samples
    for i in 0..5.min(samples.nrows()) {
        print!("Sample {}: [", i + 1);
        for j in 0..samples.ncols() {
            print!("{}", samples[(i, j)]);
            if j < samples.ncols() - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    assert_eq!(samples.nrows(), 15, "Should produce 15 unique samples");
}
