//! Check Rust's summation behavior

#[test]
fn check_rust_summation() {
    let tau1: f64 = 0.43795963770742691;
    let tau2: f64 = 0.56204036229257304;

    let r_sum: f64 = tau1 + tau2;

    println!("tau1 = {:.17}", tau1);
    println!("tau2 = {:.17}", tau2);
    println!("r_sum = {:.17}", r_sum);
    println!("r_sum == 1.0: {}", r_sum == 1.0);
    println!("r_sum - 1.0 = {:.17e}", r_sum - 1.0);
    println!("Bit representation of r_sum: {:016x}", r_sum.to_bits());
    println!("Bit representation of 1.0:   {:016x}", 1.0f64.to_bits());

    // Try with slice sum (like nalgebra might use)
    let vec = vec![tau1, tau2];
    let r_iter_sum: f64 = vec.iter().sum();
    println!("\nUsing iter().sum():");
    println!("sum = {:.17}", r_iter_sum);
    println!("sum == 1.0: {}", r_iter_sum == 1.0);
    println!("Bit representation: {:016x}", r_iter_sum.to_bits());
}
