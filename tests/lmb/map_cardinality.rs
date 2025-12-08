//! Test MAP cardinality estimation against MATLAB values
use multisensor_lmb_filters_rs::lmb::cardinality::{
    elementary_symmetric_function, lmb_map_cardinality_estimate,
};

#[test]
fn test_map_cardinality_seed42_t64() {
    // These are the actual r values from debug output at seed 42, t=64
    // MATLAB has 10 objects with these existence probabilities
    let r_matlab = vec![
        0.999629, 0.999640, 0.999613, 0.999623, 0.999569, 0.999589, 0.997505, 0.999208, 0.998461,
        0.997084,
    ];

    println!("\n=== Testing MAP Cardinality for Seed 42, t=64 ===");
    println!("Input r values (MATLAB):");
    for (i, &ri) in r_matlab.iter().enumerate() {
        println!("  r[{}] = {:.6}", i, ri);
    }

    // Test elementary symmetric function
    let r_adjusted: Vec<f64> = r_matlab.iter().map(|&ri| ri - 1e-6).collect();
    let mut r_ratio = Vec::new();
    let mut prod_1_minus_r = 1.0;
    for &ri in &r_adjusted {
        prod_1_minus_r *= 1.0 - ri;
        r_ratio.push(ri / (1.0 - ri));
    }

    println!("\nAdjusted r (r - 1e-6):");
    for (i, &ri) in r_adjusted.iter().enumerate() {
        println!("  r_adj[{}] = {:.10}", i, ri);
    }

    println!("\nRatio r/(1-r):");
    for (i, &ratio) in r_ratio.iter().enumerate() {
        println!("  ratio[{}] = {:.6}", i, ratio);
    }

    println!("\nprod(1-r) = {:.15e}", prod_1_minus_r);

    let esf = elementary_symmetric_function(&r_ratio);

    println!("\nESF values:");
    for (i, &e) in esf.iter().enumerate() {
        println!("  esf[{}] = {:.15e}", i, e);
    }

    let rho: Vec<f64> = esf.iter().map(|&e| prod_1_minus_r * e).collect();

    println!("\nRho values (cardinality distribution):");
    for (i, &rho_i) in rho.iter().enumerate() {
        println!("  rho[{}] = {:.15e}", i, rho_i);
    }

    // Find maximum
    let (max_idx, &max_val) = rho
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("\nMaximum rho:");
    println!("  index = {} (this is n_map before capping)", max_idx);
    println!("  value = {:.15e}", max_val);

    let n_map = std::cmp::min(max_idx, r_matlab.len());
    println!("\nn_map (capped to length) = {}", n_map);
    println!("Expected from MATLAB: 10");

    // Also test the full function
    let (n_map_full, indices) = lmb_map_cardinality_estimate(&r_matlab);
    println!("\nFull function result:");
    println!("  n_map = {}", n_map_full);
    println!("  indices = {:?}", indices);

    println!("\n=== MATLAB Expected Results ===");
    println!("  n_map = 10");
    println!("  indices = [2, 1, 4, 3, 6, 5, 8, 9, 7, 10] (1-indexed)");
    println!("  indices = [1, 0, 3, 2, 5, 4, 7, 8, 6, 9] (0-indexed)");
}
