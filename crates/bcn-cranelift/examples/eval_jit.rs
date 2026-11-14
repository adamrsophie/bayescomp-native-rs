//! Like `bcn-core`'s `eval`, through the Cranelift JIT. Exits with status 2
//! if any result is not bit identical to the reference interpreter.

use bcn_core::{exec::Runtime, lower::lower};
use bcn_cranelift::Compiled;

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
    let jit = Compiled::new(&prog).expect("compile");
    let mut rt = Runtime::new(prog.clone());
    let mut ri = Runtime::new(prog);
    let mut same = true;
    for line in std::fs::read_to_string(&a[4]).unwrap().lines() {
        let u: Vec<f64> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        for (jac, pro) in [(true, false), (false, false), (true, true), (false, true)] {
            let mut g = vec![0.0; dim];
            let mut gi = vec![0.0; dim];
            let r = jit.eval(&mut rt, &u, jac, pro, Some(&mut g));
            let ri_ = ri.interpret(&u, jac, pro, Some(&mut gi));
            match (&r, &ri_) {
                (Ok(v), Ok(w)) => same &= v.to_bits() == w.to_bits() && g.iter().zip(&gi).all(|(x, y)| x.to_bits() == y.to_bits()),
                (Err(e), Err(f)) => same &= e == f,
                _ => same = false,
            }
            match r {
                Ok(v) => {
                    let gs: Vec<String> = g.iter().map(|x| format!("{x:.17e}")).collect();
                    println!("{v:.17e} {}", gs.join(" "));
                }
                Err(e) => println!("ERR {e}"),
            }
        }
    }
    if !same {
        eprintln!("JIT and interpreter differ");
        std::process::exit(2);
    }
}
