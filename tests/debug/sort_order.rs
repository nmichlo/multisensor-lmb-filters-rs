//! Debug the sorting order issue

#[test]
fn debug_sort_order() {
    // These are the actual r values from t=64
    let r = vec![
        1.0,
        0.835414564460917,
        0.999904411345262,
        1.0,
        0.478570605023971,
        1.0,
        1.0,
        1.0,
        0.999513735410475,
        0.962171419493074,
        0.079371559379718,
        0.069073639335945,
    ];

    // Adjust r
    let r_adjusted: Vec<f64> = r.iter().map(|&ri| ri - 1e-6).collect();

    println!("\nOriginal r values:");
    for (i, &ri) in r.iter().enumerate() {
        println!("  r[{}] = {:.15}", i, ri);
    }

    println!("\nAdjusted r values:");
    for (i, &ri) in r_adjusted.iter().enumerate() {
        println!("  r_adj[{}] = {:.15}", i, ri);
    }

    // Check if r=1.0 values are truly equal after adjustment
    println!("\nObjects with r=1.0:");
    let ones: Vec<usize> = r.iter().enumerate()
        .filter(|(_, &ri)| ri == 1.0)
        .map(|(i, _)| i)
        .collect();
    println!("  Indices: {:?}", ones);

    println!("\nAdjusted values for r=1.0 objects:");
    for &i in &ones {
        println!("  r_adj[{}] = {:.15} (bits: {:064b})", i, r_adjusted[i], r_adjusted[i].to_bits());
    }

    // Sort with stable secondary key
    let mut indexed_r: Vec<(usize, f64)> = r_adjusted.iter().enumerate().map(|(i, &val)| (i, val)).collect();
    indexed_r.sort_by(|(i_a, a), (i_b, b)| {
        match b.partial_cmp(a).unwrap() {
            std::cmp::Ordering::Equal => i_a.cmp(i_b),
            other => other,
        }
    });

    println!("\nSorted indices:");
    for (rank, (idx, val)) in indexed_r.iter().enumerate() {
        println!("  Rank {}: index={}, value={:.15}", rank, idx, val);
    }

    let sorted_indices: Vec<usize> = indexed_r.iter().map(|(i, _)| *i).collect();
    println!("\nFinal sorted indices: {:?}", sorted_indices);
    println!("Expected from MATLAB: [0, 3, 5, 6, 7, 2, 8, 9, 1]");
}
