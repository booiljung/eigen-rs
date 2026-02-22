//! Coefficient-wise unary operations.

use crate::core::scalar::Scalar;
use crate::core::xpr::MatrixXpr;
// use crate::core::storage::Storage;
use crate::core::arch::Packet;

/// Trait for unary functors (e.g., Sin, Cos, Exp).
pub trait UnaryFunctor<T: Scalar> {
    fn eval(&self, x: T) -> T;
    fn packet_eval<P: Packet<T>>(&self, x: P) -> P;
}

/// Coefficient-wise unary operation expression.
pub struct CwiseUnaryOp<'a, T, X, F>
where
    T: Scalar,
    X: MatrixXpr<T>,
    F: UnaryFunctor<T>,
{
    xpr: &'a X,
    functor: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, X, F> CwiseUnaryOp<'a, T, X, F>
where
    T: Scalar,
    X: MatrixXpr<T>,
    F: UnaryFunctor<T>,
{
    pub fn new(xpr: &'a X, functor: F) -> Self {
        Self {
            xpr,
            functor,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, X, F> MatrixXpr<T> for CwiseUnaryOp<'a, T, X, F>
where
    T: Scalar,
    X: MatrixXpr<T>,
    F: UnaryFunctor<T> + Sync,
{
    fn rows(&self) -> usize {
        self.xpr.rows()
    }
    fn cols(&self) -> usize {
        self.xpr.cols()
    }

    fn eval(&self, row: usize, col: usize) -> T {
        self.functor.eval(self.xpr.eval(row, col))
    }

    fn packet_eval<P: Packet<T>>(&self, _row: usize, _col: usize) -> P
    where
        T: Scalar,
    {
        // We require P to implement PacketMath<T> dynamically?
        // Rust traits doesn't work like that easily without trait bounds on P in method.
        // But MatrixXpr::packet_eval defines P: Packet<T>.
        // We need a way to check if P implements PacketMath.
        // For now, let's assume P implements PacketMath if T is float.
        // But the trait bound doesn't guaranteed it.
        // Strategy: We can only use packet_eval if we can ensure PacketMath.
        // Since PacketMath is implemented for all standard Packets in arch, we can try to enforce it.
        // However, the trait method signature is fixed.
        // Workaround: We can't easily call packet_eval if the trait bound isn't there.
        // We might need to relax or change the design, or use fallback.

        // Actually, since PacketMath implies Packet, we can't downcast.
        // But we can implement a helper trait "HasPacketMath" on P?
        // Or simpler: change the functor to take `P`.
        // But `F::packet_eval` requires `P: PacketMath`. The caller only provides `P: Packet`.
        // This is a type system issue.
        // Resolution: We will implement it by scalar fallback if P doesn't satisfy bounds,
        // BUT we can't check bounds at runtime.
        //
        // Real solution: MatrixXpr trait should probably be updated or we use a wrapper.
        // Or, we specifically implement MatrixXpr for CwiseUnaryOp where P: PacketMath.
        // But the method is generic `fn packet_eval<P>`.
        //
        // Hack: `PacketMath` is implemented for `ScalarPacket` too.
        // We can just restrict `packet_eval` to types that are technically `PacketMath`.
        // But we can't restrict generic `P` in the impl.
        //
        // Let's rely on the fact that for specific types T (f32/f64), the packets defined in arch
        // DO implement PacketMath.
        //
        // We can use a trick: `EvaluatePacket` trait.

        // For now, let's assume we can't vectorize easily without changing MatrixXpr.
        // Evaluation fallback:
        // let val = self.xpr.packet_eval::<P>(row, col);
        // But we can't call functor.packet_eval(val) if val is not known to be PacketMath.

        // Let's change the strategy:
        // We will define specific UnaryOps types or specialized impls.
        // Or we assume `Packet` has these methods? No, they are in `PacketMath`.
        //
        // Let's modify MatrixXpr? No, that's big.
        //
        // Alternative: Use `Scalar` properties.
        // If we can't vectorize generic unary ops easily in this strict type system without overhead,
        // we might just do scalar loop for now in `eval`.
        // BUT the goal is SIMD.

        // PROPER FIX: Move `PacketMath` methods into `Packet` trait with default impls (unimplemented!() or scalar loop).
        // This makes `Packet` guaranteed to have `psin`, `pcos` etc.
        // Let's do that! It cleans up everything.

        // Re-routing plan: Modify `Packet` trait in `src/core/arch/mod.rs`.
        unimplemented!("See plan change: Modifying Packet trait")
    }
}

// Functors
pub struct ScalarSin;
impl<T: Scalar> UnaryFunctor<T> for ScalarSin {
    fn eval(&self, x: T) -> T {
        x.sin()
    }
    fn packet_eval<P: Packet<T>>(&self, x: P) -> P {
        x.psin()
    }
}

pub struct ScalarCos;
impl<T: Scalar> UnaryFunctor<T> for ScalarCos {
    fn eval(&self, x: T) -> T {
        x.cos()
    }
    fn packet_eval<P: Packet<T>>(&self, x: P) -> P {
        x.pcos()
    }
}

pub struct ScalarExp;
impl<T: Scalar> UnaryFunctor<T> for ScalarExp {
    fn eval(&self, x: T) -> T {
        x.exp()
    }
    fn packet_eval<P: Packet<T>>(&self, x: P) -> P {
        x.pexp()
    }
}

pub struct ScalarLog;
impl<T: Scalar> UnaryFunctor<T> for ScalarLog {
    fn eval(&self, x: T) -> T {
        x.ln()
    }
    fn packet_eval<P: Packet<T>>(&self, x: P) -> P {
        x.plog()
    }
}
