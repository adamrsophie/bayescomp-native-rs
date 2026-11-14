//! Evaluate a lowered model: `eval IR.json DATA.json DIM U.txt`, where each
//! line of U.txt holds one unconstrained point. Prints one line per point and
//! (jacobian, propto) combination: `lp g1 g2 ...` or `ERR message`.

use bcn_core::{exec::Runtime, lower::lower};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ir: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    let dim: usize = a[3].parse().unwrap();
    let prog = match lower(&ir, &data, dim) {
        Ok(p) => p,
        Err(e) => {
            println!("LOWER {e}");
            return;
        }
    };
    let mut rt = Runtime::new(prog);
    for line in std::fs::read_to_string(&a[4]).unwrap().lines() {
        let u: Vec<f64> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        for (jac, pro) in [(true, false), (false, false), (true, true), (false, true)] {
            let mut g = vec![0.0; dim];
            match rt.interpret(&u, jac, pro, Some(&mut g)) {
                Ok(v) => {
                    let gs: Vec<String> = g.iter().map(|x| format!("{x:.17e}")).collect();
                    println!("{v:.17e} {}", gs.join(" "));
                }
                Err(e) => println!("ERR {e}"),
            }
        }
    }
}
