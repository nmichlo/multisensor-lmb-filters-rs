use multisensor_lmb_filters_rs::lmb::multisensor::{
    MultisensorAssociator, MultisensorGibbsAssociator,
};
use multisensor_lmb_filters_rs::AssociationConfig;
use serde_json;
use std::fs;

#[test]
fn test_gibbs_debug() {
    // Load fixture
    let fixture_path = "tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json";
    let fixture_str = fs::read_to_string(fixture_path).expect("Failed to read fixture");
    let fixture: serde_json::Value =
        serde_json::from_str(&fixture_str).expect("Failed to parse fixture");

    // Extract L as nested arrays, then flatten
    let l_nested: Vec<Vec<Vec<f64>>> =
        serde_json::from_value(fixture["step3_gibbs"]["input"]["L"].clone())
            .expect("Failed to parse L");

    // Flatten in row-major order (C order)
    let mut log_likelihoods = Vec::new();
    for i in &l_nested {
        for j in i {
            for &k in j {
                log_likelihoods.push(k);
            }
        }
    }

    // Dimensions are [m1+1, m2+1, ..., ms+1, n] where mi is number of measurements from sensor i
    // For this fixture: 2 sensors with [2, 11] measurements, 4 objects
    let dimensions = vec![3, 12, 4]; // [2+1, 11+1, 4]

    let mut rng = rand::thread_rng();

    let associator = MultisensorGibbsAssociator;
    let config = AssociationConfig {
        gibbs_samples: 1000,
        ..Default::default()
    };

    let result = associator
        .associate(&mut rng, &log_likelihoods, &dimensions, &config)
        .expect("Association failed");

    eprintln!("\n=== FINAL RESULT ===");
    eprintln!("Number of unique samples: {}", result.num_samples());
    eprintln!("All samples:");
    for (i, sample) in result.samples.iter().enumerate() {
        eprintln!("  Sample {}: {:?}", i + 1, sample);
    }

    // MATLAB produces 15 unique samples
    assert_eq!(
        result.num_samples(),
        15,
        "Should produce 15 unique samples like MATLAB"
    );
}
