//! Minimal benchmark runner for native Rust.
//!
//! Usage:
//!     benchmark_single --scenario <path> --filter <name>
//!
//! Output:
//!     Prints elapsed time in milliseconds as a single number.
//!     Exit 0 on success, non-zero on error.

use clap::Parser;

use multisensor_lmb_filters_rs::bench_utils::{create_filter, load_scenario, preprocess};

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser)]
#[command(name = "benchmark_single")]
#[command(about = "Minimal benchmark runner for native Rust")]
struct Args {
    /// Path to scenario JSON file
    #[arg(long)]
    scenario: String,

    /// Filter name (e.g., LMB-LBP, LMBM-Gibbs, AA-LMB-LBP)
    #[arg(long)]
    filter: String,

    /// Print filter configuration JSON (can combine with --skip-run)
    #[arg(long)]
    get_config: bool,

    /// Skip running the benchmark (useful with --get-config)
    #[arg(long)]
    skip_run: bool,
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load and preprocess scenario
    let scenario = load_scenario(&args.scenario);
    let prep = preprocess(&scenario);

    // Create filter (single source of truth from bench_utils)
    let mut filter = match create_filter(&args.filter, &prep) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    // Output config if requested
    if args.get_config {
        println!("{}", filter.get_config_json());
    }

    // Run benchmark unless --skip-run
    if !args.skip_run {
        let (avg_ms, std_ms) = filter.run(&prep);
        println!("{:.4},{:.4}", avg_ms, std_ms);
    }

    Ok(())
}
