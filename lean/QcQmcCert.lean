/-
  QcQmcCert.lean
  Self-contained, zero-dependency Lean 4 formalization (Level L2: zero sorry, standard axioms only)
  for:
    "Holomorphic Obstructions and Surrogate-Debiased Finite-Sample Spectral Certificates
     for Machine-Learned Quantum Monte Carlo"

  Every theorem below is proved parametrically over an arbitrary linearly ordered field `OrdField K`
  (covering both `ℝ` and `ℚ`), with zero `sorry` and zero custom axioms.

  Formalized results (16 audited symbols in `AuditAxioms.lean`):
    1. `linear_projector_collapse`              (Remark 2.3: left-projector collapse G0 + P A)
    2. `plucker_kernel_distance_min`            (Theorem 2.4: real orthogonal-projection distance floor)
    3. `plucker_kernel_distance_attained`       (Theorem 2.4: exact attainment by projection)
    4. `plucker_kernel_distance_min_complex`    (Theorem 2.4: ComplexK projection distance floor)
    5. `mixed_estimator_bias_decomp`            (Theorem 2.5: exact mixed-estimator bias split)
    6. `holomorphic_zero_bias`                  (Theorem 2.5: zero bias when r = 0 & exact eigenstate)
    7. `nonholomorphic_bias_persists`           (Theorem 2.5: persistent residual bias on eigenstate)
    8. `variational_boost_spectral_bound`       (Theorem 3.1: universal 2x2 PSD spectral-edge bound)
    9. `variational_boost_hamiltonian`          (Theorem 3.1: universal Mixed-Energy Variational-Boost Law)
   10. `variational_boost_sharp`                (Theorem 3.1: exact sharpness on rank-1 boundary)
   11. `block_hausdorff_edge_soundness`         (Theorem 4.1(i): real single-edge conic-dual soundness)
   12. `block_hausdorff_gap_soundness`          (Theorem 4.1(ii): real two-edge gap conic-dual soundness)
   13. `block_hausdorff_edge_soundness_complex` (Theorem 4.1(i): ComplexK single-edge dual soundness)
   14. `block_hausdorff_gap_soundness_complex`  (Theorem 4.1(ii): ComplexK two-edge gap dual soundness)
   15. `linear_box_upper_bound`                 (Theorem 4.1: worst-case box transfer)
   16. `ppi_control_variate_unbiased`           (Proposition 4.2(ii): finite-population PPI expectation identity)
-/

namespace QcQmcCert

-- ============================================================================
-- 0. Linearly Ordered Commutative Field Interface
-- ============================================================================

/-- Standard axioms of a linearly ordered commutative field (satisfied by ℝ and ℚ). -/
class OrdField (K : Type) extends Add K, Sub K, Mul K, Div K, Neg K, Zero K, One K, LT K, LE K where
  add_comm : ∀ a b : K, a + b = b + a
  add_assoc : ∀ a b c : K, (a + b) + c = a + (b + c)
  zero_add : ∀ a : K, 0 + a = a
  add_neg_cancel : ∀ a : K, a + (-a) = 0
  sub_eq_add_neg : ∀ a b : K, a - b = a + (-b)
  mul_comm : ∀ a b : K, a * b = b * a
  mul_assoc : ∀ a b c : K, (a * b) * c = a * (b * c)
  one_mul : ∀ a : K, 1 * a = a
  left_distrib : ∀ a b c : K, a * (b + c) = a * b + a * c
  div_eq_mul_inv : ∀ a b : K, a / b = a * (1 / b)
  mul_inv_cancel : ∀ a : K, a ≠ 0 → a * (1 / a) = 1
  le_refl : ∀ a : K, a ≤ a
  le_trans : ∀ {a b c : K}, a ≤ b → b ≤ c → a ≤ c
  le_antisymm : ∀ {a b : K}, a ≤ b → b ≤ a → a = b
  le_of_lt : ∀ {a b : K}, a < b → a ≤ b
  lt_of_lt_of_le : ∀ {a b c : K}, a < b → b ≤ c → a < c
  lt_of_le_of_lt : ∀ {a b c : K}, a ≤ b → b < c → a < c
  lt_irrefl : ∀ a : K, ¬ (a < a)
  le_or_gt : ∀ a b : K, a ≤ b ∨ b < a
  add_le_add : ∀ {a b c d : K}, a ≤ b → c ≤ d → a + c ≤ b + d
  add_lt_add_right : ∀ {a b : K}, a < b → ∀ c : K, a + c < b + c
  mul_nonneg : ∀ {a b : K}, 0 ≤ a → 0 ≤ b → 0 ≤ a * b
  mul_pos : ∀ {a b : K}, 0 < a → 0 < b → 0 < a * b
  mul_le_mul_of_nonneg_left : ∀ {a b c : K}, a ≤ b → 0 ≤ c → c * a ≤ c * b
  mul_le_mul_of_nonneg_right : ∀ {a b c : K}, a ≤ b → 0 ≤ c → a * c ≤ b * c
  mul_lt_mul_of_pos_right : ∀ {a b c : K}, a < b → 0 < c → a * c < b * c
  sq_nonneg : ∀ a : K, 0 ≤ a * a
  sq_pos_of_ne_zero : ∀ {a : K}, a ≠ 0 → 0 < a * a
  div_pos : ∀ {a b : K}, 0 < a → 0 < b → 0 < a / b
  zero_le_one : (0 : K) ≤ 1
  sub_nonneg_of_le : ∀ {a b : K}, a ≤ b → 0 ≤ b - a
  le_of_sub_nonneg : ∀ {a b : K}, 0 ≤ b - a → a ≤ b
  sub_pos_of_lt : ∀ {a b : K}, a < b → 0 < b - a
  lt_of_sub_pos : ∀ {a b : K}, 0 < b - a → a < b
  absVal : K → K
  le_abs_self : ∀ a : K, a ≤ absVal a
  neg_le_abs_self : ∀ a : K, -a ≤ absVal a
  abs_nonneg : ∀ a : K, 0 ≤ absVal a

section CoreAlgebra
variable {K : Type} [OrdField K]

theorem add_zero' (a : K) : a + 0 = a := by
  rw [OrdField.add_comm, OrdField.zero_add]

theorem mul_one' (a : K) : a * 1 = a := by
  rw [OrdField.mul_comm, OrdField.one_mul]

theorem right_distrib' (a b c : K) : (a + b) * c = a * c + b * c := by
  rw [OrdField.mul_comm, OrdField.left_distrib, OrdField.mul_comm c a, OrdField.mul_comm c b]

theorem mul_zero' (a : K) : a * 0 = 0 := by
  have h1 : a * 0 + a * 0 = a * 0 := by
    rw [← OrdField.left_distrib, OrdField.zero_add]
  have h2 := congrArg (· + -(a * 0)) h1
  rw [OrdField.add_assoc, OrdField.add_neg_cancel, add_zero'] at h2
  exact h2

theorem zero_mul' (a : K) : 0 * a = 0 := by
  rw [OrdField.mul_comm, mul_zero']

theorem neg_add_cancel' (a : K) : (-a) + a = 0 := by
  rw [OrdField.add_comm, OrdField.add_neg_cancel]

theorem mul_neg' (a b : K) : a * (-b) = -(a * b) := by
  have h : a * b + a * (-b) = 0 := by
    rw [← OrdField.left_distrib, OrdField.add_neg_cancel, mul_zero']
  have h2 := congrArg (-(a * b) + ·) h
  rw [← OrdField.add_assoc, neg_add_cancel', OrdField.zero_add, add_zero'] at h2
  exact h2

theorem neg_mul' (a b : K) : (-a) * b = -(a * b) := by
  rw [OrdField.mul_comm, mul_neg', OrdField.mul_comm]

theorem neg_neg' (a : K) : -(-a) = a := by
  have h : (-a) + a = 0 := neg_add_cancel' a
  have h2 := congrArg (· + -(-a)) h
  rw [OrdField.add_assoc, OrdField.add_comm a (-(-a)), ← OrdField.add_assoc,
      OrdField.add_neg_cancel, OrdField.zero_add, OrdField.zero_add] at h2
  exact h2.symm

theorem mul_sub' (a b c : K) : a * (b - c) = a * b - a * c := by
  rw [OrdField.sub_eq_add_neg, OrdField.sub_eq_add_neg, OrdField.left_distrib, mul_neg']

theorem sub_mul' (a b c : K) : (a - b) * c = a * c - b * c := by
  rw [OrdField.sub_eq_add_neg, OrdField.sub_eq_add_neg, right_distrib', neg_mul']

theorem sub_self' (a : K) : a - a = 0 := by
  rw [OrdField.sub_eq_add_neg, OrdField.add_neg_cancel]

theorem sub_zero' (a : K) : a - 0 = a := by
  have h0 : -(0 : K) = 0 := by
    have := OrdField.add_neg_cancel (0 : K)
    rw [OrdField.zero_add] at this
    exact this
  rw [OrdField.sub_eq_add_neg, h0, add_zero']

theorem sub_add_cancel' (a b : K) : (a - b) + b = a := by
  rw [OrdField.sub_eq_add_neg, OrdField.add_assoc, neg_add_cancel', add_zero']

theorem add_sub_cancel' (a b : K) : (a + b) - b = a := by
  rw [OrdField.sub_eq_add_neg, OrdField.add_assoc, OrdField.add_neg_cancel, add_zero']

theorem neg_add' (a b : K) : -(a + b) = (-a) + (-b) := by
  have h : (a + b) + ((-a) + (-b)) = 0 := by
    calc (a + b) + ((-a) + (-b))
        = (a + (-a)) + (b + (-b)) := by
            rw [OrdField.add_assoc, ← OrdField.add_assoc b (-a) (-b),
                OrdField.add_comm b (-a), OrdField.add_assoc (-a) b (-b),
                ← OrdField.add_assoc a (-a)]
      _ = 0 + 0 := by rw [OrdField.add_neg_cancel, OrdField.add_neg_cancel]
      _ = 0 := OrdField.zero_add 0
  have h2 := congrArg (-(a + b) + ·) h
  rw [← OrdField.add_assoc, neg_add_cancel', OrdField.zero_add, add_zero'] at h2
  exact h2.symm

theorem div_mul_cancel' (a b : K) (hb : b ≠ 0) : (a / b) * b = a := by
  rw [OrdField.div_eq_mul_inv, OrdField.mul_assoc, OrdField.mul_comm (1 / b) b,
      OrdField.mul_inv_cancel b hb, mul_one']

theorem div_add_div_same (a b c : K) : a / c + b / c = (a + b) / c := by
  rw [OrdField.div_eq_mul_inv a c, OrdField.div_eq_mul_inv b c,
      OrdField.div_eq_mul_inv (a + b) c, right_distrib']

theorem div_sub_div_same (a b c : K) : a / c - b / c = (a - b) / c := by
  rw [OrdField.div_eq_mul_inv a c, OrdField.div_eq_mul_inv b c,
      OrdField.div_eq_mul_inv (a - b) c, sub_mul']

theorem mul_div_cancel_right' (a b : K) (hb : b ≠ 0) : (a * b) / b = a := by
  rw [OrdField.div_eq_mul_inv, OrdField.mul_assoc, OrdField.mul_inv_cancel b hb, mul_one']

theorem ne_zero_of_pos {a : K} (h : 0 < a) : a ≠ 0 := by
  intro heq
  rw [heq] at h
  exact OrdField.lt_irrefl 0 h

theorem sub_le_sub_left' {a b : K} (c : K) (h : a ≤ b) : c - b ≤ c - a := by
  apply OrdField.le_of_sub_nonneg
  have hrew : (c - a) - (c - b) = b - a := by
    rw [OrdField.sub_eq_add_neg (c - a), OrdField.sub_eq_add_neg c b,
        neg_add', neg_neg', OrdField.sub_eq_add_neg c a,
        OrdField.add_assoc c (-a), OrdField.add_comm (-a) (-c + b),
        ← OrdField.add_assoc c (-c + b), ← OrdField.add_assoc c (-c) b,
        OrdField.add_neg_cancel, OrdField.zero_add, ← OrdField.sub_eq_add_neg]
  rw [hrew]
  exact OrdField.sub_nonneg_of_le h

theorem sub_lt_sub_right' {a b : K} (c : K) (h : a < b) : a - c < b - c := by
  rw [OrdField.sub_eq_add_neg, OrdField.sub_eq_add_neg]
  exact OrdField.add_lt_add_right h (-c)

theorem le_add_of_nonneg_right {a b : K} (hb : 0 ≤ b) : a ≤ a + b := by
  have h := OrdField.add_le_add (OrdField.le_refl a) hb
  rw [add_zero'] at h
  exact h

theorem neg_le_neg' {a b : K} (h : a ≤ b) : -b ≤ -a := by
  have h0 := sub_le_sub_left' (0 : K) h
  rw [OrdField.sub_eq_add_neg, OrdField.zero_add, OrdField.sub_eq_add_neg, OrdField.zero_add] at h0
  exact h0

theorem mul_nonneg_of_nonpos_of_nonpos {a b : K} (ha : a ≤ 0) (hb : b ≤ 0) : 0 ≤ a * b := by
  have h1 : 0 ≤ -a := by
    have := OrdField.sub_nonneg_of_le ha
    rw [OrdField.sub_eq_add_neg, OrdField.zero_add] at this
    exact this
  have h2 : 0 ≤ -b := by
    have := OrdField.sub_nonneg_of_le hb
    rw [OrdField.sub_eq_add_neg, OrdField.zero_add] at this
    exact this
  have h3 := OrdField.mul_nonneg h1 h2
  rw [neg_mul', mul_neg', neg_neg'] at h3
  exact h3

/-- Finite list sum in an ordered field. -/
def listSum : List K → K
  | [] => 0
  | x :: xs => x + listSum xs

theorem listSum_map_add {α : Type} (l : List α) (f g : α → K) :
    listSum (l.map (fun x => f x + g x)) = listSum (l.map f) + listSum (l.map g) := by
  induction l with
  | nil => simp [listSum, OrdField.zero_add]
  | cons a t ih =>
    simp only [List.map_cons, listSum, ih]
    rw [OrdField.add_assoc (f a) (g a), ← OrdField.add_assoc (g a),
        OrdField.add_comm (g a) (listSum (t.map f)), OrdField.add_assoc (listSum (t.map f)),
        ← OrdField.add_assoc (f a)]

theorem listSum_map_sub {α : Type} (l : List α) (f g : α → K) :
    listSum (l.map (fun x => f x - g x)) = listSum (l.map f) - listSum (l.map g) := by
  induction l with
  | nil => simp [listSum, sub_self']
  | cons a t ih =>
    simp only [List.map_cons, listSum, ih]
    rw [OrdField.sub_eq_add_neg (f a), OrdField.sub_eq_add_neg (listSum (t.map f)),
        OrdField.sub_eq_add_neg (f a + listSum (t.map f)), neg_add',
        OrdField.add_assoc (f a) (-g a), ← OrdField.add_assoc (-g a),
        OrdField.add_comm (-g a) (listSum (t.map f)), OrdField.add_assoc (listSum (t.map f)),
        ← OrdField.add_assoc (f a)]

theorem listSum_map_mul_left {α : Type} (l : List α) (c : K) (f : α → K) :
    listSum (l.map (fun x => c * f x)) = c * listSum (l.map f) := by
  induction l with
  | nil => simp [listSum, mul_zero']
  | cons a t ih =>
    simp only [List.map_cons, listSum, ih, OrdField.left_distrib]

theorem listSum_map_mul_right {α : Type} (l : List α) (f : α → K) (c : K) :
    listSum (l.map (fun x => f x * c)) = listSum (l.map f) * c := by
  induction l with
  | nil => simp [listSum, zero_mul']
  | cons a t ih =>
    simp only [List.map_cons, listSum, ih, right_distrib']

theorem listSum_map_zero {α : Type} (l : List α) :
    listSum (l.map (fun _ => (0 : K))) = 0 := by
  induction l with
  | nil => rfl
  | cons _ t ih => simp [listSum, ih, OrdField.zero_add]

theorem listSum_nonneg {α : Type} (l : List α) (f : α → K)
    (h : ∀ x ∈ l, 0 ≤ f x) : 0 ≤ listSum (l.map f) := by
  induction l with
  | nil => exact OrdField.le_refl 0
  | cons a t ih =>
    simp only [List.map_cons, listSum]
    have ha : 0 ≤ f a := h a (List.Mem.head t)
    have ht : 0 ≤ listSum (t.map f) := ih (fun x hx => h x (List.Mem.tail a hx))
    have hsum := OrdField.add_le_add ha ht
    rw [OrdField.zero_add] at hsum
    exact hsum

theorem listSum_le_listSum {α : Type} (l : List α) (f g : α → K)
    (h : ∀ x ∈ l, f x ≤ g x) : listSum (l.map f) ≤ listSum (l.map g) := by
  induction l with
  | nil => exact OrdField.le_refl 0
  | cons a t ih =>
    simp only [List.map_cons, listSum]
    exact OrdField.add_le_add (h a (List.Mem.head t)) (ih (fun x hx => h x (List.Mem.tail a hx)))

theorem listSum_swap {α β : Type} (la : List α) (lb : List β) (f : α → β → K) :
    listSum (la.map (fun a => listSum (lb.map (fun b => f a b)))) =
    listSum (lb.map (fun b => listSum (la.map (fun a => f a b)))) := by
  induction la with
  | nil =>
    simp only [List.map_nil, listSum]
    symm
    exact listSum_map_zero lb
  | cons a t ih =>
    simp only [List.map_cons, listSum, ih]
    rw [← listSum_map_add]

/-- Natural-number power in an ordered field. -/
def powK (a : K) : Nat → K
  | 0 => 1
  | n + 1 => a * powK a n

theorem powK_add (a : K) (m n : Nat) : powK a (m + n) = powK a m * powK a n := by
  induction m with
  | zero => simp [powK, OrdField.one_mul]
  | succ k ih =>
    have hsucc : k + 1 + n = (k + n) + 1 := by omega
    rw [hsucc, powK, ih, powK, OrdField.mul_assoc]

theorem powK_nonneg {a : K} (ha : 0 ≤ a) : ∀ n : Nat, 0 ≤ powK a n
  | 0 => OrdField.zero_le_one
  | n + 1 => OrdField.mul_nonneg ha (powK_nonneg ha n)

theorem powK_le_one {a : K} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) : ∀ n : Nat, powK a n ≤ 1
  | 0 => OrdField.le_refl 1
  | n + 1 => by
    have hn := powK_le_one ha0 ha1 n
    have h1 := OrdField.mul_le_mul_of_nonneg_left hn ha0
    rw [mul_one'] at h1
    exact OrdField.le_trans h1 ha1

end CoreAlgebra

-- ============================================================================
-- 1. REMARK 1: Linear-Projector Collapse of Affine Backflow
-- ============================================================================

section LinearProjectorCollapse

/-- Minimal linear-algebraic structure capturing adjoint-action multiplication. -/
structure MatrixAlg (M : Type) where
  add : M → M → M
  mul : M → M → M
  adj : M → M
  adj_add : ∀ X Y, adj (add X Y) = add (adj X) (adj Y)
  adj_mul : ∀ X Y, adj (mul X Y) = mul (adj Y) (adj X)
  mul_assoc : ∀ X Y Z, mul (mul X Y) Z = mul X (mul Y Z)
  right_distrib : ∀ X Y Z, mul (add X Y) Z = add (mul X Z) (mul Y Z)

/-- **Remark 2.3 (Left-Projector Collapse):**
    Any left-projector affine backflow orbital map `G(P) = G₀ + P A` with Hermitian idempotent
    projector `P = P†` satisfying `P Φ = Φ` collapses identically to the constant Slater
    determinant orbitals `G₀ + A` inside `G(P)† Φ`. -/
theorem linear_projector_collapse {M : Type} (alg : MatrixAlg M)
    (G0 A P Phi : M)
    (hP_herm : alg.adj P = P)
    (hP_Phi : alg.mul P Phi = Phi) :
    alg.mul (alg.adj (alg.add G0 (alg.mul P A))) Phi =
    alg.mul (alg.adj (alg.add G0 A)) Phi := by
  rw [alg.adj_add, alg.adj_mul, hP_herm, alg.right_distrib, alg.mul_assoc, hP_Phi,
      alg.adj_add, alg.right_distrib]

end LinearProjectorCollapse

-- ============================================================================
-- 2. THEOREM 2A: Exact Plucker-Kernel Distance-to-Physical-States Floor
-- ============================================================================

section PluckerDistance
variable {K : Type} [OrdField K]

/-- Inner product and squared norm on finite-dimensional vectors indexed by a list. -/
def dotProd {ι : Type} (idx : List ι) (u v : ι → K) : K :=
  listSum (idx.map (fun i => u i * v i))

def normSq {ι : Type} (idx : List ι) (u : ι → K) : K :=
  dotProd idx u u

theorem normSq_nonneg {ι : Type} (idx : List ι) (u : ι → K) : 0 ≤ normSq idx u :=
  listSum_nonneg idx (fun i => u i * u i) (fun i _ => OrdField.sq_nonneg (u i))

theorem normSq_add_orthogonal {ι : Type} (idx : List ι) (x y : ι → K)
    (horth : dotProd idx x y = 0) :
    normSq idx (fun i => x i + y i) = normSq idx x + normSq idx y := by
  unfold normSq dotProd
  have hexp : ∀ i, (x i + y i) * (x i + y i) =
      (x i * x i + y i * y i) + (x i * y i + x i * y i) := by
    intro i
    rw [OrdField.left_distrib, right_distrib', right_distrib', OrdField.mul_comm (y i) (x i)]
    -- (xx + yx) + (xy + yy) = (xx + yy) + (xy + xy)
    rw [OrdField.add_assoc (x i * x i), ← OrdField.add_assoc (x i * y i) (x i * y i) (y i * y i),
        OrdField.add_comm (x i * y i + x i * y i) (y i * y i),
        ← OrdField.add_assoc (x i * x i) (y i * y i)]
  have hmap : idx.map (fun i => (x i + y i) * (x i + y i)) =
      idx.map (fun i => (x i * x i + y i * y i) + (x i * y i + x i * y i)) := by
    congr; funext i; exact hexp i
  rw [hmap, listSum_map_add, listSum_map_add, listSum_map_add]
  change (listSum (idx.map (fun i => x i * x i)) + listSum (idx.map (fun i => y i * y i))) +
         (dotProd idx x y + dotProd idx x y) = _
  rw [horth, OrdField.zero_add, add_zero']

/-- **Theorem 2A (Exact Plücker-Kernel Distance Floor):**
    Let `g = g_par + g_perp` where `g_par` lies in the physical overlap subspace `V` and
    `g_perp` is orthogonal to every `v ∈ V` (in particular `g_perp = (I - S S⁺) g`).
    Then for EVERY physical state overlap vector `v ∈ V`,
      `‖g - v‖² = ‖g_perp‖² + ‖g_par - v‖² ≥ ‖g_perp‖²`. -/
theorem plucker_kernel_distance_min {ι : Type} (idx : List ι)
    (g g_par g_perp v : ι → K)
    (hdecomp : ∀ i, g i = g_par i + g_perp i)
    (horth : dotProd idx g_perp (fun i => g_par i - v i) = 0) :
    normSq idx g_perp ≤ normSq idx (fun i => g i - v i) := by
  have hrew : (fun i => g i - v i) = (fun i => g_perp i + (g_par i - v i)) := by
    funext i
    rw [hdecomp i, OrdField.sub_eq_add_neg, OrdField.sub_eq_add_neg,
        OrdField.add_comm (g_par i) (g_perp i), OrdField.add_assoc]
  rw [hrew, normSq_add_orthogonal idx g_perp (fun i => g_par i - v i) horth]
  exact le_add_of_nonneg_right (normSq_nonneg idx (fun i => g_par i - v i))

/-- **Theorem 2A (Exact Attainment):** Equality holds at `v = g_par`. -/
theorem plucker_kernel_distance_attained {ι : Type} (idx : List ι)
    (g g_par g_perp : ι → K)
    (hdecomp : ∀ i, g i = g_par i + g_perp i) :
    normSq idx (fun i => g i - g_par i) = normSq idx g_perp := by
  have hrew : (fun i => g i - g_par i) = g_perp := by
    funext i
    rw [hdecomp i, OrdField.add_comm (g_par i) (g_perp i), add_sub_cancel']
  rw [hrew]

end PluckerDistance

-- ============================================================================
-- 3. THEOREM 2B: Exact Mixed-Estimator Bias Decomposition
-- ============================================================================

section MixedEstimatorBias
variable {K : Type} [OrdField K]

/-- **Theorem 2B (Exact Mixed-Estimator Bias Decomposition):**
    For any walker ensemble `(w_i, f_i)` with nonzero overlaps `f_i ≠ 0`, total weight
    `W = ∑ w_i ≠ 0`, decomposition `f_i = f_phys_i + r_i`, and linear local-energy numerator
    `LH_f_i = LH_phys_i + LH_r_i`:
      `(∑ (w_i / f_i) LH_f_i) / W - E0`
      `= (∑ (w_i / f_i)(LH_phys_i - E0 f_phys_i)) / W + (∑ (w_i / f_i)(LH_r_i - E0 r_i)) / W`. -/
theorem mixed_estimator_bias_decomp {ι : Type} (walkers : List ι)
    (w f f_phys r LH_f LH_phys LH_r : ι → K) (E0 : K)
    (hf_nz : ∀ i ∈ walkers, f i ≠ 0)
    (hf_dec : ∀ i, f i = f_phys i + r i)
    (hLH_dec : ∀ i, LH_f i = LH_phys i + LH_r i)
    (hW_nz : listSum (walkers.map w) ≠ 0) :
    listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / listSum (walkers.map w) - E0 =
    listSum (walkers.map (fun i => (w i / f i) * (LH_phys i - E0 * f_phys i))) / listSum (walkers.map w) +
    listSum (walkers.map (fun i => (w i / f i) * (LH_r i - E0 * r i))) / listSum (walkers.map w) := by
  let W := listSum (walkers.map w)
  change listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / W - E0 =
         listSum (walkers.map (fun i => (w i / f i) * (LH_phys i - E0 * f_phys i))) / W +
         listSum (walkers.map (fun i => (w i / f i) * (LH_r i - E0 * r i))) / W
  have hE0_W : E0 = (E0 * W) / W := (mul_div_cancel_right' E0 W hW_nz).symm
  have hLHS : listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / W - E0 =
              (listSum (walkers.map (fun i => (w i / f i) * LH_f i)) - E0 * W) / W := by
    calc listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / W - E0
        = listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / W - (E0 * W) / W :=
          congrArg (listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / W - ·) hE0_W
      _ = (listSum (walkers.map (fun i => (w i / f i) * LH_f i)) - E0 * W) / W :=
          div_sub_div_same _ _ W
  rw [hLHS, div_add_div_same]
  congr 1
  -- Show E0 * W = ∑ (w_i / f_i) * (E0 * f_i)
  have hW_exp : E0 * W = listSum (walkers.map (fun i => (w i / f i) * (E0 * f i))) := by
    rw [← listSum_map_mul_left]
    have hmap : walkers.map (fun i => E0 * w i) =
                walkers.map (fun i => (w i / f i) * (E0 * f i)) := by
      apply List.map_congr_left
      intro i hi
      calc E0 * w i = E0 * ((w i / f i) * f i) := by rw [div_mul_cancel' (w i) (f i) (hf_nz i hi)]
        _ = (w i / f i) * (E0 * f i) := by
            rw [← OrdField.mul_assoc, OrdField.mul_comm E0 (w i / f i), OrdField.mul_assoc]
    exact congrArg listSum hmap
  rw [hW_exp, ← listSum_map_sub, ← listSum_map_add]
  congr 1
  apply List.map_congr_left
  intro i _
  rw [hLH_dec i, hf_dec i, ← mul_sub', ← OrdField.left_distrib]
  congr 1
  rw [OrdField.left_distrib E0 (f_phys i) (r i),
      OrdField.sub_eq_add_neg (LH_phys i + LH_r i), neg_add',
      OrdField.sub_eq_add_neg (LH_phys i), OrdField.sub_eq_add_neg (LH_r i),
      OrdField.add_assoc (LH_phys i) (LH_r i),
      ← OrdField.add_assoc (LH_r i) (-(E0 * f_phys i)),
      OrdField.add_comm (LH_r i) (-(E0 * f_phys i)),
      OrdField.add_assoc (-(E0 * f_phys i)) (LH_r i),
      ← OrdField.add_assoc (LH_phys i)]

/-- **Corollary (Holomorphic Unbiasedness):** When `r ≡ 0` and the walkers represent an exact
    eigenstate (`∑ (w_i/f_i)(LH_phys_i - E0 f_phys_i) = 0`), the mixed estimator equals `E0`. -/
theorem holomorphic_zero_bias {ι : Type} (walkers : List ι)
    (w f f_phys LH_f LH_phys : ι → K) (E0 : K)
    (hf_nz : ∀ i ∈ walkers, f i ≠ 0)
    (hf_eq : ∀ i, f i = f_phys i)
    (hLH_eq : ∀ i, LH_f i = LH_phys i)
    (hW_nz : listSum (walkers.map w) ≠ 0)
    (hGS : listSum (walkers.map (fun i => (w i / f i) * (LH_phys i - E0 * f_phys i))) = 0) :
    listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / listSum (walkers.map w) - E0 = 0 := by
  have h := mixed_estimator_bias_decomp walkers w f f_phys (fun _ => 0) LH_f LH_phys (fun _ => 0) E0
    hf_nz (fun i => by rw [hf_eq i, add_zero']) (fun i => by rw [hLH_eq i, add_zero']) hW_nz
  rw [h, hGS]
  have hr0 : listSum (walkers.map (fun i => (w i / f i) * ((0 : K) - E0 * 0))) = 0 := by
    have hm : walkers.map (fun i => (w i / f i) * ((0 : K) - E0 * 0)) =
              walkers.map (fun _ => (0 : K)) := by
      apply List.map_congr_left
      intro i _
      rw [mul_zero', sub_self', mul_zero']
    rw [hm]
    exact listSum_map_zero walkers
  rw [hr0, OrdField.div_eq_mul_inv 0, zero_mul', OrdField.zero_add]

/-- **Corollary (Persistent Non-Holomorphic Bias on Exact Ground State):**
    Even with zero walker error (`∑ (w_i/f_i)(LH_phys_i - E0 f_phys_i) = 0`), any non-holomorphic
    residual `r` leaves the exact bias `(∑ (w_i/f_i)(LH_r_i - E0 r_i)) / W`. -/
theorem nonholomorphic_bias_persists {ι : Type} (walkers : List ι)
    (w f f_phys r LH_f LH_phys LH_r : ι → K) (E0 : K)
    (hf_nz : ∀ i ∈ walkers, f i ≠ 0)
    (hf_dec : ∀ i, f i = f_phys i + r i)
    (hLH_dec : ∀ i, LH_f i = LH_phys i + LH_r i)
    (hW_nz : listSum (walkers.map w) ≠ 0)
    (hGS : listSum (walkers.map (fun i => (w i / f i) * (LH_phys i - E0 * f_phys i))) = 0) :
    listSum (walkers.map (fun i => (w i / f i) * LH_f i)) / listSum (walkers.map w) - E0 =
    listSum (walkers.map (fun i => (w i / f i) * (LH_r i - E0 * r i))) / listSum (walkers.map w) := by
  have h := mixed_estimator_bias_decomp walkers w f f_phys r LH_f LH_phys LH_r E0
    hf_nz hf_dec hLH_dec hW_nz
  rw [h, hGS, OrdField.div_eq_mul_inv 0, zero_mul', OrdField.zero_add]

end MixedEstimatorBias

-- ============================================================================
-- 4. THEOREM 3: Universal Mixed-Energy Variational-Boost Law & Sharpness
-- ============================================================================

section VariationalBoost
variable {K : Type} [OrdField K]

/-- Core 2×2 PSD spectral-edge lower bound parameterized directly by nonnegative `β2`
    (requiring no square-root existence over an arbitrary ordered field `K`, e.g. `ℚ`). -/
theorem variational_boost_spectral_bound_sq (α β2 γ lam : K)
    (hα : 0 < α)
    (hpsd : β2 ≤ α * γ)
    (h_lam_a : α ≤ lam)
    (h_lam_g : γ ≤ lam)
    (hdet : β2 ≤ (lam - α) * (lam - γ)) :
    α + β2 / α ≤ lam := by
  have hα_nz : α ≠ 0 := ne_zero_of_pos hα
  rcases OrdField.le_or_gt (α + β2 / α) lam with hle | hlt
  · exact hle
  · exfalso
    have h1 : lam - α < β2 / α := by
      have ht := sub_lt_sub_right' α hlt
      rw [OrdField.add_comm α (β2 / α), add_sub_cancel'] at ht
      exact ht
    have hγ_lb : β2 / α ≤ γ := by
      have hpos : 0 < 1 / α := OrdField.div_pos (by
        have h1pos : (0 : K) < 1 := by
          have := OrdField.sq_pos_of_ne_zero (a := (1 : K)) (by
            intro h10
            have := congrArg (· * α) h10
            rw [OrdField.one_mul, zero_mul'] at this
            exact hα_nz this)
          rw [mul_one'] at this
          exact this
        exact h1pos) hα
      have hm := OrdField.mul_le_mul_of_nonneg_right hpsd (OrdField.le_of_lt hpos)
      rw [← OrdField.div_eq_mul_inv, OrdField.mul_comm α γ,
          ← OrdField.div_eq_mul_inv, mul_div_cancel_right' γ α hα_nz] at hm
      exact hm
    have h2 : lam - γ < α := by
      have hstep1 : lam - γ ≤ lam - β2 / α := sub_le_sub_left' lam hγ_lb
      have hstep2 : lam - β2 / α < α := by
        have ht := sub_lt_sub_right' (β2 / α) hlt
        rw [add_sub_cancel'] at ht
        exact ht
      exact OrdField.lt_of_le_of_lt hstep1 hstep2
    have h_la_nn : 0 ≤ lam - α := OrdField.sub_nonneg_of_le h_lam_a
    have h_lg_nn : 0 ≤ lam - γ := OrdField.sub_nonneg_of_le h_lam_g
    have hprod1 : (lam - α) * (lam - γ) ≤ (β2 / α) * (lam - γ) :=
      OrdField.mul_le_mul_of_nonneg_right (OrdField.le_of_lt h1) h_lg_nn
    have hβ2_pos : 0 < β2 / α := OrdField.lt_of_le_of_lt h_la_nn h1
    have hprod2 : (β2 / α) * (lam - γ) < (β2 / α) * α := by
      rw [OrdField.mul_comm (β2 / α) (lam - γ), OrdField.mul_comm (β2 / α) α]
      exact OrdField.mul_lt_mul_of_pos_right h2 hβ2_pos
    rw [div_mul_cancel' β2 α hα_nz] at hprod2
    have hstrict : (lam - α) * (lam - γ) < β2 := OrdField.lt_of_le_of_lt hprod1 hprod2
    exact OrdField.lt_irrefl β2 (OrdField.lt_of_le_of_lt hdet hstrict)

/-- **Theorem 3.1 (Core 2×2 PSD Spectral-Edge Lower Bound):**
    Let `M = [[α, β], [β, γ]] ≽ 0` with `α > 0` and `α γ ≥ β²`, and suppose `λ I - M ≽ 0`
    (i.e. `α ≤ λ`, `γ ≤ λ`, and `(λ - α)(λ - γ) ≥ β²`). Then
      `λ ≥ α + β² / α`. -/
theorem variational_boost_spectral_bound (α β γ lam : K)
    (hα : 0 < α)
    (hpsd : β * β ≤ α * γ)
    (h_lam_a : α ≤ lam)
    (h_lam_g : γ ≤ lam)
    (hdet : β * β ≤ (lam - α) * (lam - γ)) :
    α + (β * β) / α ≤ lam :=
  variational_boost_spectral_bound_sq α (β * β) γ lam hα hpsd h_lam_a h_lam_g hdet

/-- **Theorem 3.1 (Hamiltonian Form — Universal Mixed-Energy Variational-Boost Law):**
    Substituting `λ = E_max - E_0`, `α = E_max - E_sur > 0`, and `β² = d_H² / (1 - b₀²)`
    (no square-root assumption required over any ordered field `K`):
      `E_0 ≤ E_sur - (d_H² / (1 - b₀²)) / (E_max - E_sur)`. -/
theorem variational_boost_hamiltonian (E0 Esur Emax dH2 oneMinusB0Sq γ : K)
    (hgap : 0 < Emax - Esur)
    (_hb : 0 < oneMinusB0Sq)
    (hpsd : dH2 / oneMinusB0Sq ≤ (Emax - Esur) * γ)
    (hE0_sur : Emax - Esur ≤ Emax - E0)
    (hE0_γ : γ ≤ Emax - E0)
    (hdet : dH2 / oneMinusB0Sq ≤ ((Emax - E0) - (Emax - Esur)) * ((Emax - E0) - γ)) :
    E0 ≤ Esur - (dH2 / oneMinusB0Sq) / (Emax - Esur) := by
  have hmain := variational_boost_spectral_bound_sq (Emax - Esur) (dH2 / oneMinusB0Sq) γ (Emax - E0)
    hgap hpsd hE0_sur hE0_γ hdet
  have hsub := OrdField.sub_nonneg_of_le hmain
  apply OrdField.le_of_sub_nonneg
  have hrew : (Esur - (dH2 / oneMinusB0Sq) / (Emax - Esur)) - E0 =
              (Emax - E0) - ((Emax - Esur) + (dH2 / oneMinusB0Sq) / (Emax - Esur)) := by
    let term := (dH2 / oneMinusB0Sq) / (Emax - Esur)
    change (Esur - term) - E0 = (Emax - E0) - ((Emax - Esur) + term)
    rw [OrdField.sub_eq_add_neg (Emax - E0), neg_add',
        OrdField.sub_eq_add_neg Emax Esur, neg_add', neg_neg',
        OrdField.sub_eq_add_neg Emax E0,
        OrdField.sub_eq_add_neg (Esur - term),
        OrdField.sub_eq_add_neg Esur term]
    calc Esur + -term + -E0
        = (Emax + -Emax) + (Esur + -term + -E0) := by
            rw [OrdField.add_neg_cancel, OrdField.zero_add]
      _ = (Emax + -E0) + ((-Emax + Esur) + -term) := by
            rw [OrdField.add_assoc Emax (-Emax),
                OrdField.add_comm (Esur + -term) (-E0),
                ← OrdField.add_assoc (-Emax) (-E0) (Esur + -term),
                OrdField.add_comm (-Emax) (-E0),
                OrdField.add_assoc (-E0) (-Emax) (Esur + -term),
                ← OrdField.add_assoc Emax (-E0),
                ← OrdField.add_assoc (-Emax) Esur (-term)]
  rw [hrew]
  exact hsub

/-- **Theorem 3 (Exact Sharpness):** At the rank-1 boundary `γ = β² / α`, the top eigenvalue
    `λ = α + β² / α` attains exact equality in the characteristic equation `(λ - α)(λ - γ) = β²`. -/
theorem variational_boost_sharp (α β : K) (hα : α ≠ 0) :
    let γ := (β * β) / α
    let lam := α + γ
    (lam - α) * (lam - γ) = β * β := by
  intro γ lam
  have h1 : lam - α = γ := by
    dsimp only [lam]
    rw [OrdField.add_comm α γ, add_sub_cancel']
  have h2 : lam - γ = α := by
    dsimp only [lam]
    exact add_sub_cancel' α γ
  rw [h1, h2]
  exact div_mul_cancel' (β * β) α hα

end VariationalBoost

-- ============================================================================
-- 5. THEOREM 4: Surrogate-Anchored Block Hausdorff Conic-Dual Soundness
--               (Single-Edge E0 AND Two-Edge Spectral Gap (θ₁, θ₂))
-- ============================================================================

section BlockHausdorffSoundness
variable {K : Type} [OrdField K]

/-- A discrete spectral atom `(t_s, u_s, v_s)` where `t_s` is an eigenvalue of `T` and
    `u_s = ⟨Ψ_sur | E_s⟩`, `v_s = ⟨Ψ_T | E_s⟩`. -/
structure SpecAtom (K : Type) where
  t : K
  u : K
  v : K

/-- Block moments `(a_k, b_k, c_k)` generated by a discrete spectral measure `atoms`. -/
def momA (atoms : List (SpecAtom K)) (k : Nat) : K :=
  listSum (atoms.map (fun s => (s.u * s.u) * powK s.t k))

def momB (atoms : List (SpecAtom K)) (k : Nat) : K :=
  listSum (atoms.map (fun s => (s.u * s.v) * powK s.t k))

def momC (atoms : List (SpecAtom K)) (k : Nat) : K :=
  listSum (atoms.map (fun s => (s.v * s.v) * powK s.t k))

/-- A rank-1 factor of a 2(d+1)×2(d+1) PSD dual matrix is a list of pairs `(i, x0, x1)`
    representing the polynomial pair `p(t) = ∑ (x0_i u + x1_i v) t^i`. -/
structure PolyTerm (K : Type) where
  deg : Nat
  x0 : K
  x1 : K

/-- Evaluate the linear form `∑ (x0 u + x1 v) t^i` at an atom `s`. -/
def evalVec (terms : List (PolyTerm K)) (s : SpecAtom K) : K :=
  listSum (terms.map (fun r => (r.x0 * s.u + r.x1 * s.v) * powK s.t r.deg))

/-- Rank-1 shifted block-Hankel contraction `xᵀ H^(shift)(a, b, c) x` computed purely from
    the moment sequences `(a, b, c)`. -/
def rank1MomentForm (a b c : Nat → K) (shift : Nat) (terms : List (PolyTerm K)) : K :=
  listSum (terms.map (fun r1 =>
    listSum (terms.map (fun r2 =>
      let k := r1.deg + r2.deg + shift
      r1.x0 * r2.x0 * a k + (r1.x0 * r2.x1 + r1.x1 * r2.x0) * b k + r1.x1 * r2.x1 * c k))))

/-- Core Factorization Identity:
    `xᵀ H^(shift)(momA, momB, momC) x = ∑_s t_s^{shift} · (evalVec x s)²`. -/
theorem rank1MomentForm_eq_weighted_sum_sq (atoms : List (SpecAtom K))
    (shift : Nat) (terms : List (PolyTerm K)) :
    rank1MomentForm (momA atoms) (momB atoms) (momC atoms) shift terms =
    listSum (atoms.map (fun s => powK s.t shift * (evalVec terms s * evalVec terms s))) := by
  unfold rank1MomentForm momA momB momC
  -- Step 1: express inner (r1, r2) term as sum over atoms
  have hinner : ∀ r1 r2 : PolyTerm K,
      let k := r1.deg + r2.deg + shift
      r1.x0 * r2.x0 * listSum (atoms.map (fun s => (s.u * s.u) * powK s.t k)) +
      (r1.x0 * r2.x1 + r1.x1 * r2.x0) * listSum (atoms.map (fun s => (s.u * s.v) * powK s.t k)) +
      r1.x1 * r2.x1 * listSum (atoms.map (fun s => (s.v * s.v) * powK s.t k)) =
      listSum (atoms.map (fun s =>
        powK s.t shift *
        (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
         ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))) := by
    intro r1 r2
    dsimp only
    rw [← listSum_map_mul_left, ← listSum_map_mul_left, ← listSum_map_mul_left,
        ← listSum_map_add, ← listSum_map_add]
    congr 1
    apply List.map_congr_left
    intro s _
    -- Factor powK s.t (r1.deg + r2.deg + shift) = powK s.t shift * powK s.t r1.deg * powK s.t r2.deg
    have hpow : powK s.t (r1.deg + r2.deg + shift) =
                powK s.t shift * (powK s.t r1.deg * powK s.t r2.deg) := by
      rw [powK_add, powK_add, OrdField.mul_comm (powK s.t r1.deg * powK s.t r2.deg) (powK s.t shift)]
    rw [hpow]
    let P := powK s.t shift
    let p1 := powK s.t r1.deg
    let p2 := powK s.t r2.deg
    let t1 := r1.x0 * s.u + r1.x1 * s.v
    let t2 := r2.x0 * s.u + r2.x1 * s.v
    have hprod : t1 * t2 =
        r1.x0 * r2.x0 * (s.u * s.u) + (r1.x0 * r2.x1 + r1.x1 * r2.x0) * (s.u * s.v) +
        r1.x1 * r2.x1 * (s.v * s.v) := by
      dsimp only [t1, t2]
      rw [OrdField.left_distrib, right_distrib', right_distrib', right_distrib']
      have m1 : r1.x0 * s.u * (r2.x0 * s.u) = r1.x0 * r2.x0 * (s.u * s.u) := by
        rw [OrdField.mul_assoc, ← OrdField.mul_assoc s.u r2.x0 s.u,
            OrdField.mul_comm s.u r2.x0, OrdField.mul_assoc r2.x0 s.u s.u, ← OrdField.mul_assoc]
      have m2 : r1.x1 * s.v * (r2.x0 * s.u) = r1.x1 * r2.x0 * (s.u * s.v) := by
        rw [OrdField.mul_assoc, ← OrdField.mul_assoc s.v r2.x0 s.u,
            OrdField.mul_comm s.v r2.x0, OrdField.mul_assoc r2.x0 s.v s.u,
            OrdField.mul_comm s.v s.u, ← OrdField.mul_assoc]
      have m3 : r1.x0 * s.u * (r2.x1 * s.v) = r1.x0 * r2.x1 * (s.u * s.v) := by
        rw [OrdField.mul_assoc, ← OrdField.mul_assoc s.u r2.x1 s.v,
            OrdField.mul_comm s.u r2.x1, OrdField.mul_assoc r2.x1 s.u s.v, ← OrdField.mul_assoc]
      have m4 : r1.x1 * s.v * (r2.x1 * s.v) = r1.x1 * r2.x1 * (s.v * s.v) := by
        rw [OrdField.mul_assoc, ← OrdField.mul_assoc s.v r2.x1 s.v,
            OrdField.mul_comm s.v r2.x1, OrdField.mul_assoc r2.x1 s.v s.v, ← OrdField.mul_assoc]
      rw [m1, m2, m3, m4]
      let A := r1.x0 * r2.x0 * (s.u * s.u)
      let B := r1.x0 * r2.x1 * (s.u * s.v)
      let C := r1.x1 * r2.x0 * (s.u * s.v)
      let D := r1.x1 * r2.x1 * (s.v * s.v)
      change (A + C) + (B + D) = A + (B + C) + D
      rw [OrdField.add_assoc A C (B + D), ← OrdField.add_assoc C B D,
          OrdField.add_comm C B, ← OrdField.add_assoc A (B + C) D]
    -- Let's avoid fragile intermediate rewrites by doing a clean calc from scratch!
    have hprod_clean :
        r1.x0 * r2.x0 * (s.u * s.u * (P * (p1 * p2))) +
        (r1.x0 * r2.x1 + r1.x1 * r2.x0) * (s.u * s.v * (P * (p1 * p2))) +
        r1.x1 * r2.x1 * (s.v * s.v * (P * (p1 * p2))) =
        P * (t1 * p1 * (t2 * p2)) := by
      have hfactor :
          r1.x0 * r2.x0 * (s.u * s.u * (P * (p1 * p2))) +
          (r1.x0 * r2.x1 + r1.x1 * r2.x0) * (s.u * s.v * (P * (p1 * p2))) +
          r1.x1 * r2.x1 * (s.v * s.v * (P * (p1 * p2))) =
          (r1.x0 * r2.x0 * (s.u * s.u) + (r1.x0 * r2.x1 + r1.x1 * r2.x0) * (s.u * s.v) +
           r1.x1 * r2.x1 * (s.v * s.v)) * (P * (p1 * p2)) := by
        symm
        rw [right_distrib', right_distrib',
            OrdField.mul_assoc (r1.x0 * r2.x0) (s.u * s.u) (P * (p1 * p2)),
            OrdField.mul_assoc (r1.x0 * r2.x1 + r1.x1 * r2.x0) (s.u * s.v) (P * (p1 * p2)),
            OrdField.mul_assoc (r1.x1 * r2.x1) (s.v * s.v) (P * (p1 * p2))]
      rw [hfactor, ← hprod]
      -- Goal: t1 * t2 * (P * (p1 * p2)) = P * (t1 * p1 * (t2 * p2))
      rw [OrdField.mul_comm (t1 * t2) (P * (p1 * p2)), OrdField.mul_assoc P (p1 * p2) (t1 * t2)]
      congr 1
      -- Goal: p1 * p2 * (t1 * t2) = t1 * p1 * (t2 * p2)
      rw [OrdField.mul_assoc, ← OrdField.mul_assoc p2 t1 t2, OrdField.mul_comm p2 t1,
          OrdField.mul_assoc t1 p2 t2, ← OrdField.mul_assoc p1 t1 (p2 * t2),
          OrdField.mul_comm p1 t1, OrdField.mul_comm p2 t2]
    exact hprod_clean
  -- Substitute hinner and swap summations
  have h1 : listSum (terms.map (fun r1 =>
      listSum (terms.map (fun r2 =>
        let k := r1.deg + r2.deg + shift
        r1.x0 * r2.x0 * listSum (atoms.map (fun s => (s.u * s.u) * powK s.t k)) +
        (r1.x0 * r2.x1 + r1.x1 * r2.x0) * listSum (atoms.map (fun s => (s.u * s.v) * powK s.t k)) +
        r1.x1 * r2.x1 * listSum (atoms.map (fun s => (s.v * s.v) * powK s.t k)))))) =
    listSum (terms.map (fun r1 =>
      listSum (terms.map (fun r2 =>
        listSum (atoms.map (fun s =>
          powK s.t shift *
          (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
           ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))))))) := by
    congr 1; apply List.map_congr_left; intro r1 _
    congr 1; apply List.map_congr_left; intro r2 _
    exact hinner r1 r2
  rw [h1]
  -- Swap sums: ∑_r1 ∑_r2 ∑_s = ∑_s ∑_r1 ∑_r2
  have h2 : ∀ r1 : PolyTerm K,
      listSum (terms.map (fun r2 =>
        listSum (atoms.map (fun s =>
          powK s.t shift *
          (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
           ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))))) =
      listSum (atoms.map (fun s =>
        listSum (terms.map (fun r2 =>
          powK s.t shift *
          (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
           ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))))) := by
    intro r1; exact listSum_swap terms atoms _
  have h2_all : listSum (terms.map (fun r1 =>
      listSum (terms.map (fun r2 =>
        listSum (atoms.map (fun s =>
          powK s.t shift *
          (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
           ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))))))) =
    listSum (terms.map (fun r1 =>
      listSum (atoms.map (fun s =>
        listSum (terms.map (fun r2 =>
          powK s.t shift *
          (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
           ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))))))) := by
    congr 1; apply List.map_congr_left; intro r1 _; exact h2 r1
  rw [h2_all, listSum_swap terms atoms]
  congr 1
  apply List.map_congr_left
  intro s _
  -- Show ∑_r1 ∑_r2 P * (f(r1) * f(r2)) = P * (evalVec * evalVec)
  unfold evalVec
  have hf : ∀ r1 : PolyTerm K,
      listSum (terms.map (fun r2 =>
        powK s.t shift *
        (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
         ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))) =
      powK s.t shift *
        (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
         listSum (terms.map (fun r2 => (r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg))) := by
    intro r1
    rw [listSum_map_mul_left, listSum_map_mul_left]
  have hf_all :
      listSum (terms.map (fun r1 =>
        listSum (terms.map (fun r2 =>
          powK s.t shift *
          (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
           ((r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg)))))) =
      listSum (terms.map (fun r1 =>
        powK s.t shift *
          (((r1.x0 * s.u + r1.x1 * s.v) * powK s.t r1.deg) *
           listSum (terms.map (fun r2 => (r2.x0 * s.u + r2.x1 * s.v) * powK s.t r2.deg))))) := by
    congr 1; apply List.map_congr_left; intro r1 _; exact hf r1
  rw [hf_all, listSum_map_mul_left, listSum_map_mul_right]

/-- A PSD dual matrix encoded as a finite sum of rank-1 outer products (e.g. from exact LDLᵀ). -/
abbrev PSDMat (K : Type) := List (List (PolyTerm K))

def psdMomentForm (a b c : Nat → K) (shift : Nat) (M : PSDMat K) : K :=
  listSum (M.map (fun terms => rank1MomentForm a b c shift terms))

/-- Single-edge dual functional:
    `Φ_edge(P, Q, R; θ) = ⟨P, H_d⟩ + ⟨Q, H_d^(1)⟩ + ⟨R, θ H_d - H_d^(1)⟩`. -/
def edgeDualValue (P Q R : PSDMat K) (θ : K) (a b c : Nat → K) : K :=
  psdMomentForm a b c 0 P +
  psdMomentForm a b c 1 Q +
  (θ * psdMomentForm a b c 0 R - psdMomentForm a b c 1 R)

/-- Nonnegativity of `edgeDualValue` whenever all spectral atoms lie in `[0, θ]`. -/
theorem edgeDualValue_nonneg (atoms : List (SpecAtom K)) (P Q R : PSDMat K) (θ : K)
    (hsupp : ∀ s ∈ atoms, 0 ≤ s.t ∧ s.t ≤ θ) :
    0 ≤ edgeDualValue P Q R θ (momA atoms) (momB atoms) (momC atoms) := by
  unfold edgeDualValue psdMomentForm
  have hP : 0 ≤ listSum (P.map (fun terms => rank1MomentForm (momA atoms) (momB atoms) (momC atoms) 0 terms)) := by
    apply listSum_nonneg
    intro terms _
    rw [rank1MomentForm_eq_weighted_sum_sq]
    apply listSum_nonneg
    intro s _
    rw [powK, OrdField.one_mul]
    exact OrdField.sq_nonneg (evalVec terms s)
  have hQ : 0 ≤ listSum (Q.map (fun terms => rank1MomentForm (momA atoms) (momB atoms) (momC atoms) 1 terms)) := by
    apply listSum_nonneg
    intro terms _
    rw [rank1MomentForm_eq_weighted_sum_sq]
    apply listSum_nonneg
    intro s hs
    have ht0 := (hsupp s hs).1
    rw [powK, powK, mul_one']
    exact OrdField.mul_nonneg ht0 (OrdField.sq_nonneg (evalVec terms s))
  have hR : 0 ≤ θ * listSum (R.map (fun terms => rank1MomentForm (momA atoms) (momB atoms) (momC atoms) 0 terms)) -
                listSum (R.map (fun terms => rank1MomentForm (momA atoms) (momB atoms) (momC atoms) 1 terms)) := by
    rw [← listSum_map_mul_left, ← listSum_map_sub]
    apply listSum_nonneg
    intro terms _
    rw [rank1MomentForm_eq_weighted_sum_sq, rank1MomentForm_eq_weighted_sum_sq,
        ← listSum_map_mul_left, ← listSum_map_sub]
    apply listSum_nonneg
    intro s hs
    have hθs : 0 ≤ θ - s.t := OrdField.sub_nonneg_of_le (hsupp s hs).2
    rw [powK, OrdField.one_mul, powK, powK, mul_one', ← sub_mul']
    exact OrdField.mul_nonneg hθs (OrdField.sq_nonneg (evalVec terms s))
  have hPQ := OrdField.add_le_add hP hQ
  rw [OrdField.zero_add] at hPQ
  have hPQR := OrdField.add_le_add hPQ hR
  rw [OrdField.zero_add] at hPQR
  exact hPQR

/-- **Theorem 4A (Surrogate-Anchored Block Hausdorff Single-Edge Soundness):**
    If all spectral atoms satisfy `t_s ≥ 0` and a PSD triple `(P, Q, R)` has strictly negative
    dual value `edgeDualValue P Q R θ < 0`, then there MUST exist a spectral atom with `t_s > θ`. -/
theorem block_hausdorff_edge_soundness (atoms : List (SpecAtom K)) (P Q R : PSDMat K) (θ : K)
    (ht0 : ∀ s ∈ atoms, 0 ≤ s.t)
    (hneg : edgeDualValue P Q R θ (momA atoms) (momB atoms) (momC atoms) < 0) :
    ¬ (∀ s ∈ atoms, s.t ≤ θ) := by
  intro hall
  have hnn := edgeDualValue_nonneg atoms P Q R θ (fun s hs => ⟨ht0 s hs, hall s hs⟩)
  exact OrdField.lt_irrefl (0 : K) (OrdField.lt_of_le_of_lt hnn hneg)

/-- Two-edge spectral-gap dual functional:
    `Φ_gap(P, Q, R, G; θ₁, θ₂) = ⟨P, H_d⟩ + ⟨Q, H_d^(1)⟩ + ⟨R, H_d - H_d^(1)⟩
                                + ⟨G, H_d^(2) - (θ₁+θ₂) H_d^(1) + θ₁ θ₂ H_d⟩`. -/
def gapDualValue (P Q R G : PSDMat K) (θ1 θ2 : K) (a b c : Nat → K) : K :=
  psdMomentForm a b c 0 P +
  psdMomentForm a b c 1 Q +
  (psdMomentForm a b c 0 R - psdMomentForm a b c 1 R) +
  (psdMomentForm a b c 2 G - (θ1 + θ2) * psdMomentForm a b c 1 G + (θ1 * θ2) * psdMomentForm a b c 0 G)

/-- Nonnegativity of `gapDualValue` whenever `θ1 ≤ θ2` and all spectral atoms lie in `[0, θ₁] ∪ [θ₂, 1]`. -/
theorem gapDualValue_nonneg (atoms : List (SpecAtom K)) (P Q R G : PSDMat K) (θ1 θ2 : K)
    (hθ : θ1 ≤ θ2)
    (hsupp : ∀ s ∈ atoms, (0 ≤ s.t ∧ s.t ≤ 1) ∧ (s.t ≤ θ1 ∨ θ2 ≤ s.t)) :
    0 ≤ gapDualValue P Q R G θ1 θ2 (momA atoms) (momB atoms) (momC atoms) := by
  unfold gapDualValue
  have hEdge1 : 0 ≤ edgeDualValue P Q R 1 (momA atoms) (momB atoms) (momC atoms) :=
    edgeDualValue_nonneg atoms P Q R 1 (fun s hs => (hsupp s hs).1)
  unfold edgeDualValue at hEdge1
  rw [OrdField.one_mul] at hEdge1
  have hG : 0 ≤ psdMomentForm (momA atoms) (momB atoms) (momC atoms) 2 G -
                (θ1 + θ2) * psdMomentForm (momA atoms) (momB atoms) (momC atoms) 1 G +
                (θ1 * θ2) * psdMomentForm (momA atoms) (momB atoms) (momC atoms) 0 G := by
    unfold psdMomentForm
    rw [← listSum_map_mul_left, ← listSum_map_mul_left, ← listSum_map_sub, ← listSum_map_add]
    apply listSum_nonneg
    intro terms _
    rw [rank1MomentForm_eq_weighted_sum_sq, rank1MomentForm_eq_weighted_sum_sq,
        rank1MomentForm_eq_weighted_sum_sq, ← listSum_map_mul_left, ← listSum_map_mul_left,
        ← listSum_map_sub, ← listSum_map_add]
    apply listSum_nonneg
    intro s hs
    have hpoly :
        powK s.t 2 * (evalVec terms s * evalVec terms s) -
        (θ1 + θ2) * (powK s.t 1 * (evalVec terms s * evalVec terms s)) +
        θ1 * θ2 * (powK s.t 0 * (evalVec terms s * evalVec terms s)) =
        ((s.t - θ1) * (s.t - θ2)) * (evalVec terms s * evalVec terms s) := by
      let q := evalVec terms s * evalVec terms s
      change powK s.t 2 * q - (θ1 + θ2) * (powK s.t 1 * q) + θ1 * θ2 * (powK s.t 0 * q) =
             ((s.t - θ1) * (s.t - θ2)) * q
      have hp2 : powK s.t 2 = s.t * s.t := by simp only [powK, mul_one']
      have hp1 : powK s.t 1 = s.t := by simp only [powK, mul_one']
      have hp0 : powK s.t 0 = 1 := rfl
      rw [hp2, hp1, hp0, OrdField.one_mul,
          ← OrdField.mul_assoc (θ1 + θ2) s.t q, ← sub_mul', ← right_distrib']
      congr 1
      rw [sub_mul', mul_sub', mul_sub', right_distrib',
          OrdField.sub_eq_add_neg (s.t * s.t), OrdField.sub_eq_add_neg (s.t * s.t - s.t * θ2),
          OrdField.sub_eq_add_neg (s.t * s.t), OrdField.sub_eq_add_neg (θ1 * s.t),
          neg_add', neg_add', neg_neg',
          OrdField.mul_comm s.t θ2, OrdField.add_assoc (s.t * s.t),
          OrdField.add_comm (-(θ1 * s.t)) (-(θ2 * s.t)),
          OrdField.add_assoc (-(θ2 * s.t)),
          ← OrdField.add_assoc (s.t * s.t)]
    rw [hpoly]
    have hquad_nn : 0 ≤ (s.t - θ1) * (s.t - θ2) := by
      rcases (hsupp s hs).2 with hle1 | hge2
      · have hle2 : s.t ≤ θ2 := OrdField.le_trans hle1 hθ
        have ha : s.t - θ1 ≤ 0 := by
          have := OrdField.sub_nonneg_of_le hle1
          have hneg := neg_le_neg' this
          have h0 : -(0 : K) = 0 := by
            have := OrdField.add_neg_cancel (0 : K); rw [OrdField.zero_add] at this; exact this
          rw [h0, OrdField.sub_eq_add_neg, neg_add', neg_neg', OrdField.add_comm,
              ← OrdField.sub_eq_add_neg] at hneg
          exact hneg
        have hb : s.t - θ2 ≤ 0 := by
          have := OrdField.sub_nonneg_of_le hle2
          have hneg := neg_le_neg' this
          have h0 : -(0 : K) = 0 := by
            have := OrdField.add_neg_cancel (0 : K); rw [OrdField.zero_add] at this; exact this
          rw [h0, OrdField.sub_eq_add_neg, neg_add', neg_neg', OrdField.add_comm,
              ← OrdField.sub_eq_add_neg] at hneg
          exact hneg
        exact mul_nonneg_of_nonpos_of_nonpos ha hb
      · have hge1 : θ1 ≤ s.t := OrdField.le_trans hθ hge2
        exact OrdField.mul_nonneg (OrdField.sub_nonneg_of_le hge1) (OrdField.sub_nonneg_of_le hge2)
    exact OrdField.mul_nonneg hquad_nn (OrdField.sq_nonneg (evalVec terms s))
  have htot := OrdField.add_le_add hEdge1 hG
  rw [OrdField.zero_add] at htot
  exact htot

/-- **Theorem 4B (Two-Edge Spectral-Gap Dual Soundness):**
    If `θ1 ≤ θ2`, all spectral atoms lie in `[0, 1]`, and a PSD quadruple `(P, Q, R, G)` has
    strictly negative dual value `gapDualValue P Q R G θ1 θ2 < 0`, then NOT all spectral atoms
    lie in `[0, θ₁] ∪ [θ₂, 1]` (i.e. spectrum strictly inside `(θ₁, θ₂)` is certified). -/
theorem block_hausdorff_gap_soundness (atoms : List (SpecAtom K))
    (P Q R G : PSDMat K) (θ1 θ2 : K)
    (hθ : θ1 ≤ θ2)
    (h01 : ∀ s ∈ atoms, 0 ≤ s.t ∧ s.t ≤ 1)
    (hneg : gapDualValue P Q R G θ1 θ2 (momA atoms) (momB atoms) (momC atoms) < 0) :
    ¬ (∀ s ∈ atoms, s.t ≤ θ1 ∨ θ2 ≤ s.t) := by
  intro hall
  have hnn := gapDualValue_nonneg atoms P Q R G θ1 θ2 hθ (fun s hs => ⟨h01 s hs, hall s hs⟩)
  exact OrdField.lt_irrefl (0 : K) (OrdField.lt_of_le_of_lt hnn hneg)

/-- **Theorem 4C (Finite-Sample Confidence-Box Transfer):**
    For any linear functional `Φ(b, c) = const + ∑_{k ∈ ks} (cb_k b_k + cc_k c_k)` that upper-bounds
    or equals `edgeDualValue` (or `gapDualValue`), if `|b_k - bhat_k| ≤ eps_k` and `0 ≤ c_k ≤ 1`
    for all `k ∈ ks`, then
      `const + ∑ (cb_k b_k + cc_k c_k) ≤ const + ∑ (cb_k bhat_k + |cb_k| eps_k + max(0, cc_k))`.
    Hence a strictly negative worst-case box bound certifies `edgeDualValue < 0` (or `gapDualValue < 0`). -/
theorem linear_box_upper_bound (ks : List Nat)
    (cb cc b bhat eps c : Nat → K) (posPart : K → K)
    (hpos_ge : ∀ x, x ≤ posPart x)
    (hpos_nn : ∀ x, 0 ≤ posPart x)
    (hb_box : ∀ k ∈ ks, -(eps k) ≤ b k - bhat k ∧ b k - bhat k ≤ eps k)
    (heps_nn : ∀ k ∈ ks, 0 ≤ eps k)
    (hc_box : ∀ k ∈ ks, 0 ≤ c k ∧ c k ≤ 1) :
    listSum (ks.map (fun k => cb k * b k + cc k * c k)) ≤
    listSum (ks.map (fun k => cb k * bhat k + OrdField.absVal (cb k) * eps k + posPart (cc k))) := by
  apply listSum_le_listSum
  intro k hk
  -- Part 1: cb k * b k ≤ cb k * bhat k + |cb k| * eps k
  have hb_term : cb k * b k ≤ cb k * bhat k + OrdField.absVal (cb k) * eps k := by
    have hdiff : cb k * b k = cb k * bhat k + cb k * (b k - bhat k) := by
      rw [mul_sub', OrdField.add_comm (cb k * bhat k), sub_add_cancel']
    rw [hdiff]
    apply OrdField.add_le_add (OrdField.le_refl _)
    rcases OrdField.le_or_gt 0 (cb k) with hcb_nn | hcb_neg
    · have h1 := OrdField.mul_le_mul_of_nonneg_left (hb_box k hk).2 hcb_nn
      have h2 := OrdField.mul_le_mul_of_nonneg_right (OrdField.le_abs_self (cb k)) (heps_nn k hk)
      exact OrdField.le_trans h1 h2
    · have hneg_cb : 0 ≤ -(cb k) := by
        have := OrdField.sub_nonneg_of_le (OrdField.le_of_lt hcb_neg)
        rw [OrdField.sub_eq_add_neg, OrdField.zero_add] at this
        exact this
      have hneg_diff : -(b k - bhat k) ≤ eps k := by
        have := neg_le_neg' (hb_box k hk).1
        rw [neg_neg'] at this
        exact this
      have h1 := OrdField.mul_le_mul_of_nonneg_left hneg_diff hneg_cb
      rw [neg_mul', mul_neg', neg_neg'] at h1
      have h2 := OrdField.mul_le_mul_of_nonneg_right (OrdField.neg_le_abs_self (cb k)) (heps_nn k hk)
      exact OrdField.le_trans h1 h2
  -- Part 2: cc k * c k ≤ posPart (cc k)
  have hc_term : cc k * c k ≤ posPart (cc k) := by
    have h1 := OrdField.mul_le_mul_of_nonneg_right (hpos_ge (cc k)) (hc_box k hk).1
    have h2 := OrdField.mul_le_mul_of_nonneg_left (hc_box k hk).2 (hpos_nn (cc k))
    rw [mul_one'] at h2
    exact OrdField.le_trans h1 h2
  exact OrdField.add_le_add hb_term hc_term

end BlockHausdorffSoundness

-- ============================================================================
-- 6. PROPOSITION 1: Exact PPI Surrogate-Debiasing Identity
-- ============================================================================

section PPIDebiasing
variable {K : Type} [OrdField K]

/-- **Proposition 1 (Exact PPI Control-Variate Unbiasedness):**
    For any finite population of walkers `pop`, weighting `g`, arbitrary (possibly biased or
    non-holomorphic) surrogate `f`, and true overlaps `v`:
      `(∑ g_a f_a) / M + (∑ g_a (v_a - f_a)) / M = (∑ g_a v_a) / M`. -/
theorem ppi_control_variate_unbiased {α : Type} (pop : List α)
    (g f v : α → K) (M : K) :
    listSum (pop.map (fun a => g a * f a)) / M +
    listSum (pop.map (fun a => g a * (v a - f a))) / M =
    listSum (pop.map (fun a => g a * v a)) / M := by
  rw [div_add_div_same, ← listSum_map_add]
  congr 2
  apply List.map_congr_left
  intro a _
  rw [← OrdField.left_distrib, OrdField.add_comm (f a), sub_add_cancel']

end PPIDebiasing

-- ============================================================================
-- 7. COMPLEX EXTENSION OVER `OrdField K`: Complex Plücker Distance & Complex
--    Spectral-Measure Block Hausdorff Soundness (using real symmetric PSD duals)
-- ============================================================================

section ComplexExtensions
variable {K : Type} [OrdField K]

/-- Complex number `re + i im` over an ordered field `K`. -/
structure ComplexK (K : Type) where
  re : K
  im : K

def cModSq (z : ComplexK K) : K :=
  z.re * z.re + z.im * z.im

def cReProdConj (u v : ComplexK K) : K :=
  u.re * v.re + u.im * v.im

def cAdd (u v : ComplexK K) : ComplexK K :=
  ⟨u.re + v.re, u.im + v.im⟩

def cSub (u v : ComplexK K) : ComplexK K :=
  ⟨u.re - v.re, u.im - v.im⟩

def cScale (s : K) (z : ComplexK K) : ComplexK K :=
  ⟨s * z.re, s * z.im⟩

theorem cModSq_nonneg (z : ComplexK K) : 0 ≤ cModSq z := by
  unfold cModSq
  have h := OrdField.add_le_add (OrdField.sq_nonneg z.re) (OrdField.sq_nonneg z.im)
  rw [OrdField.zero_add] at h
  exact h

def cReInner {ι : Type} (idx : List ι) (u v : ι → ComplexK K) : K :=
  listSum (idx.map (fun i => cReProdConj (u i) (v i)))

def cNormSq {ι : Type} (idx : List ι) (u : ι → ComplexK K) : K :=
  listSum (idx.map (fun i => cModSq (u i)))

theorem cNormSq_nonneg {ι : Type} (idx : List ι) (u : ι → ComplexK K) : 0 ≤ cNormSq idx u :=
  listSum_nonneg idx (fun i => cModSq (u i)) (fun i _ => cModSq_nonneg (u i))

/-- **Theorem 2.4 (Complex Plücker-Kernel Distance Floor):**
    For complex overlap vectors `g = g_par + g_perp` and any physical complex overlap vector `v`
    with `Re⟨g_perp, g_par - v⟩ = 0`, `‖g - v‖² ≥ ‖g_perp‖²`. -/
theorem plucker_kernel_distance_min_complex {ι : Type} (idx : List ι)
    (g g_par g_perp v : ι → ComplexK K)
    (hdecomp : ∀ i, g i = cAdd (g_par i) (g_perp i))
    (horth : cReInner idx g_perp (fun i => cSub (g_par i) (v i)) = 0) :
    cNormSq idx g_perp ≤ cNormSq idx (fun i => cSub (g i) (v i)) := by
  unfold cNormSq
  have hstep : idx.map (fun i => cModSq (cSub (g i) (v i))) =
      idx.map (fun i => (cModSq (g_perp i) + cModSq (cSub (g_par i) (v i))) +
                        (cReProdConj (g_perp i) (cSub (g_par i) (v i)) +
                         cReProdConj (g_perp i) (cSub (g_par i) (v i)))) := by
    apply List.map_congr_left
    intro i _
    rw [hdecomp i]
    unfold cModSq cReProdConj cAdd cSub
    dsimp only
    have hre : (g_par i).re + (g_perp i).re - (v i).re =
               (g_perp i).re + ((g_par i).re - (v i).re) := by
      rw [OrdField.sub_eq_add_neg, OrdField.sub_eq_add_neg,
          OrdField.add_comm (g_par i).re (g_perp i).re, OrdField.add_assoc]
    have him : (g_par i).im + (g_perp i).im - (v i).im =
               (g_perp i).im + ((g_par i).im - (v i).im) := by
      rw [OrdField.sub_eq_add_neg, OrdField.sub_eq_add_neg,
          OrdField.add_comm (g_par i).im (g_perp i).im, OrdField.add_assoc]
    rw [hre, him]
    let xr := (g_perp i).re
    let xi := (g_perp i).im
    let yr := (g_par i).re - (v i).re
    let yi := (g_par i).im - (v i).im
    change (xr + yr) * (xr + yr) + (xi + yi) * (xi + yi) =
           ((xr * xr + xi * xi) + (yr * yr + yi * yi)) + ((xr * yr + xi * yi) + (xr * yr + xi * yi))
    have hexp : ∀ a b : K, (a + b) * (a + b) = (a * a + b * b) + (a * b + a * b) := by
      intro a b
      rw [OrdField.left_distrib, right_distrib', right_distrib', OrdField.mul_comm b a,
          OrdField.add_assoc (a * a), ← OrdField.add_assoc (a * b) (a * b) (b * b),
          OrdField.add_comm (a * b + a * b) (b * b), ← OrdField.add_assoc (a * a) (b * b)]
    rw [hexp xr yr, hexp xi yi]
    -- Rearrange ((A1 + B1) + C1) + ((A2 + B2) + C2) = ((A1 + A2) + (B1 + B2)) + ((xr*yr+xi*yi)+(xr*yr+xi*yi))
    let A1 := xr * xr; let B1 := yr * yr; let p1 := xr * yr
    let A2 := xi * xi; let B2 := yi * yi; let p2 := xi * yi
    change ((A1 + B1) + (p1 + p1)) + ((A2 + B2) + (p2 + p2)) =
           ((A1 + A2) + (B1 + B2)) + ((p1 + p2) + (p1 + p2))
    have h4 : ∀ a b c d : K, (a + b) + (c + d) = (a + c) + (b + d) := by
      intro a b c d
      rw [OrdField.add_assoc a b (c + d), ← OrdField.add_assoc b c d,
          OrdField.add_comm b c, OrdField.add_assoc c b d, ← OrdField.add_assoc a c (b + d)]
    rw [h4 (A1 + B1) (p1 + p1) (A2 + B2) (p2 + p2), h4 A1 B1 A2 B2, h4 p1 p1 p2 p2]
  rw [hstep, listSum_map_add, listSum_map_add, listSum_map_add]
  have hinner : listSum (idx.map (fun i => cReProdConj (g_perp i) (cSub (g_par i) (v i)))) = 0 := horth
  rw [hinner, OrdField.zero_add, add_zero']
  exact le_add_of_nonneg_right (cNormSq_nonneg idx (fun i => cSub (g_par i) (v i)))

/-- Complex spectral atom `(t_s, u_s, v_s)` with `u_s = ⟨Ψ_sur|e_s⟩`, `v_s = ⟨Ψ_T|e_s⟩ ∈ ℂ_K`. -/
structure SpecAtomC (K : Type) where
  t : K
  u : ComplexK K
  v : ComplexK K

def momAC (atoms : List (SpecAtomC K)) (k : Nat) : K :=
  listSum (atoms.map (fun s => cModSq s.u * powK s.t k))

def momBReC (atoms : List (SpecAtomC K)) (k : Nat) : K :=
  listSum (atoms.map (fun s => cReProdConj s.u s.v * powK s.t k))

def momCC (atoms : List (SpecAtomC K)) (k : Nat) : K :=
  listSum (atoms.map (fun s => cModSq s.v * powK s.t k))

/-- Complex spectral-measure factorization: for ANY complex states `(u_s, v_s)` and ANY real
    symmetric rank-1 factor `terms`, `rank1MomentForm (momAC) (momBReC) (momCC)` splits into
    the sum of the real-part and imaginary-part real-atom moment forms! -/
def realPartAtom (s : SpecAtomC K) : SpecAtom K := ⟨s.t, s.u.re, s.v.re⟩
def imagPartAtom (s : SpecAtomC K) : SpecAtom K := ⟨s.t, s.u.im, s.v.im⟩

theorem rank1MomentForm_complex_eq_re_plus_im (atoms : List (SpecAtomC K))
    (shift : Nat) (terms : List (PolyTerm K)) :
    rank1MomentForm (momAC atoms) (momBReC atoms) (momCC atoms) shift terms =
    rank1MomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) shift terms +
    rank1MomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) shift terms := by
  have hmA : ∀ k, momAC atoms k = momA (atoms.map realPartAtom) k + momA (atoms.map imagPartAtom) k := by
    intro k
    unfold momAC momA cModSq realPartAtom imagPartAtom
    rw [List.map_map, List.map_map, ← listSum_map_add]
    congr 1; apply List.map_congr_left; intro s _
    exact right_distrib' (s.u.re * s.u.re) (s.u.im * s.u.im) (powK s.t k)
  have hmB : ∀ k, momBReC atoms k = momB (atoms.map realPartAtom) k + momB (atoms.map imagPartAtom) k := by
    intro k
    unfold momBReC momB cReProdConj realPartAtom imagPartAtom
    rw [List.map_map, List.map_map, ← listSum_map_add]
    congr 1; apply List.map_congr_left; intro s _
    exact right_distrib' (s.u.re * s.v.re) (s.u.im * s.v.im) (powK s.t k)
  have hmC : ∀ k, momCC atoms k = momC (atoms.map realPartAtom) k + momC (atoms.map imagPartAtom) k := by
    intro k
    unfold momCC momC cModSq realPartAtom imagPartAtom
    rw [List.map_map, List.map_map, ← listSum_map_add]
    congr 1; apply List.map_congr_left; intro s _
    exact right_distrib' (s.v.re * s.v.re) (s.v.im * s.v.im) (powK s.t k)
  unfold rank1MomentForm
  rw [← listSum_map_add]
  congr 1; apply List.map_congr_left; intro r1 _
  rw [← listSum_map_add]
  congr 1; apply List.map_congr_left; intro r2 _
  dsimp only
  let k := r1.deg + r2.deg + shift
  rw [hmA k, hmB k, hmC k, OrdField.left_distrib, OrdField.left_distrib, OrdField.left_distrib]
  let a1 := r1.x0 * r2.x0 * momA (atoms.map realPartAtom) k
  let a2 := r1.x0 * r2.x0 * momA (atoms.map imagPartAtom) k
  let b1 := (r1.x0 * r2.x1 + r1.x1 * r2.x0) * momB (atoms.map realPartAtom) k
  let b2 := (r1.x0 * r2.x1 + r1.x1 * r2.x0) * momB (atoms.map imagPartAtom) k
  let c1 := r1.x1 * r2.x1 * momC (atoms.map realPartAtom) k
  let c2 := r1.x1 * r2.x1 * momC (atoms.map imagPartAtom) k
  change (a1 + a2) + (b1 + b2) + (c1 + c2) = (a1 + b1 + c1) + (a2 + b2 + c2)
  have h4 : ∀ x y z w : K, (x + y) + (z + w) = (x + z) + (y + w) := by
    intro x y z w
    rw [OrdField.add_assoc x y (z + w), ← OrdField.add_assoc y z w,
        OrdField.add_comm y z, OrdField.add_assoc z y w, ← OrdField.add_assoc x z (y + w)]
  rw [h4 a1 a2 b1 b2, h4 (a1 + b1) (a2 + b2) c1 c2]

theorem psdMomentForm_complex_eq_re_plus_im (atoms : List (SpecAtomC K))
    (shift : Nat) (M : PSDMat K) :
    psdMomentForm (momAC atoms) (momBReC atoms) (momCC atoms) shift M =
    psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) shift M +
    psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) shift M := by
  unfold psdMomentForm
  rw [← listSum_map_add]
  congr 1; apply List.map_congr_left; intro terms _
  exact rank1MomentForm_complex_eq_re_plus_im atoms shift terms

/-- **Theorem 4.1(i) (Complex Spectral-Measure Single-Edge Soundness):**
    Real symmetric PSD dual matrices `(P, Q, R)` certify arbitrary COMPLEX states
    `(Ψ_sur, Ψ_T)` using ONLY `(a_k, Re(b_k), c_k)`. -/
theorem block_hausdorff_edge_soundness_complex (atoms : List (SpecAtomC K))
    (P Q R : PSDMat K) (θ : K)
    (ht0 : ∀ s ∈ atoms, 0 ≤ s.t)
    (hneg : edgeDualValue P Q R θ (momAC atoms) (momBReC atoms) (momCC atoms) < 0) :
    ¬ (∀ s ∈ atoms, s.t ≤ θ) := by
  intro hall
  have hre : 0 ≤ edgeDualValue P Q R θ (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) := by
    apply edgeDualValue_nonneg
    intro a ha
    rcases List.mem_map.mp ha with ⟨s, hs, rfl⟩
    exact ⟨ht0 s hs, hall s hs⟩
  have him : 0 ≤ edgeDualValue P Q R θ (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) := by
    apply edgeDualValue_nonneg
    intro a ha
    rcases List.mem_map.mp ha with ⟨s, hs, rfl⟩
    exact ⟨ht0 s hs, hall s hs⟩
  have hsplit : edgeDualValue P Q R θ (momAC atoms) (momBReC atoms) (momCC atoms) =
      edgeDualValue P Q R θ (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) +
      edgeDualValue P Q R θ (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) := by
    unfold edgeDualValue
    rw [psdMomentForm_complex_eq_re_plus_im atoms 0 P,
        psdMomentForm_complex_eq_re_plus_im atoms 1 Q,
        psdMomentForm_complex_eq_re_plus_im atoms 0 R,
        psdMomentForm_complex_eq_re_plus_im atoms 1 R,
        OrdField.left_distrib]
    have h4 : ∀ x y z w : K, (x + y) + (z + w) = (x + z) + (y + w) := by
      intro x y z w
      rw [OrdField.add_assoc x y (z + w), ← OrdField.add_assoc y z w,
          OrdField.add_comm y z, OrdField.add_assoc z y w, ← OrdField.add_assoc x z (y + w)]
    have hsub : ∀ x y z w : K, (x + y) - (z + w) = (x - z) + (y - w) := by
      intro x y z w
      rw [OrdField.sub_eq_add_neg (x + y), neg_add',
          OrdField.sub_eq_add_neg x z, OrdField.sub_eq_add_neg y w, h4 x y (-z) (-w)]
    rw [hsub]
    let P1 := psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 0 P
    let P2 := psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 0 P
    let Q1 := psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 1 Q
    let Q2 := psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 1 Q
    let R1 := θ * psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 0 R -
                  psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 1 R
    let R2 := θ * psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 0 R -
                  psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 1 R
    change (P1 + P2) + (Q1 + Q2) + (R1 + R2) = (P1 + Q1 + R1) + (P2 + Q2 + R2)
    rw [h4 P1 P2 Q1 Q2, h4 (P1 + Q1) (P2 + Q2) R1 R2]
  have hnn := OrdField.add_le_add hre him
  rw [OrdField.zero_add, ← hsplit] at hnn
  exact OrdField.lt_irrefl (0 : K) (OrdField.lt_of_le_of_lt hnn hneg)

/-- **Theorem 4.1(ii) (Complex Spectral-Measure Two-Edge Spectral-Gap Soundness):**
    Real symmetric PSD dual matrices `(P, Q, R, G)` certify arbitrary COMPLEX states
    `(Ψ_sur, Ψ_T)` using ONLY `(a_k, Re(b_k), c_k)`. -/
theorem block_hausdorff_gap_soundness_complex (atoms : List (SpecAtomC K))
    (P Q R G : PSDMat K) (θ1 θ2 : K)
    (hθ : θ1 ≤ θ2)
    (h01 : ∀ s ∈ atoms, 0 ≤ s.t ∧ s.t ≤ 1)
    (hneg : gapDualValue P Q R G θ1 θ2 (momAC atoms) (momBReC atoms) (momCC atoms) < 0) :
    ¬ (∀ s ∈ atoms, s.t ≤ θ1 ∨ θ2 ≤ s.t) := by
  intro hall
  have hre : 0 ≤ gapDualValue P Q R G θ1 θ2 (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) := by
    apply gapDualValue_nonneg _ _ _ _ _ _ _ hθ
    intro a ha
    rcases List.mem_map.mp ha with ⟨s, hs, rfl⟩
    exact ⟨h01 s hs, hall s hs⟩
  have him : 0 ≤ gapDualValue P Q R G θ1 θ2 (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) := by
    apply gapDualValue_nonneg _ _ _ _ _ _ _ hθ
    intro a ha
    rcases List.mem_map.mp ha with ⟨s, hs, rfl⟩
    exact ⟨h01 s hs, hall s hs⟩
  have hsplit : gapDualValue P Q R G θ1 θ2 (momAC atoms) (momBReC atoms) (momCC atoms) =
      gapDualValue P Q R G θ1 θ2 (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) +
      gapDualValue P Q R G θ1 θ2 (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) := by
    unfold gapDualValue
    rw [psdMomentForm_complex_eq_re_plus_im atoms 0 P,
        psdMomentForm_complex_eq_re_plus_im atoms 1 Q,
        psdMomentForm_complex_eq_re_plus_im atoms 0 R,
        psdMomentForm_complex_eq_re_plus_im atoms 1 R,
        psdMomentForm_complex_eq_re_plus_im atoms 2 G,
        psdMomentForm_complex_eq_re_plus_im atoms 1 G,
        psdMomentForm_complex_eq_re_plus_im atoms 0 G,
        OrdField.left_distrib (θ1 + θ2), OrdField.left_distrib (θ1 * θ2)]
    have h4 : ∀ x y z w : K, (x + y) + (z + w) = (x + z) + (y + w) := by
      intro x y z w
      rw [OrdField.add_assoc x y (z + w), ← OrdField.add_assoc y z w,
          OrdField.add_comm y z, OrdField.add_assoc z y w, ← OrdField.add_assoc x z (y + w)]
    have hsub : ∀ x y z w : K, (x + y) - (z + w) = (x - z) + (y - w) := by
      intro x y z w
      rw [OrdField.sub_eq_add_neg (x + y), neg_add',
          OrdField.sub_eq_add_neg x z, OrdField.sub_eq_add_neg y w, h4 x y (-z) (-w)]
    rw [hsub, hsub]
    let P1 := psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 0 P
    let P2 := psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 0 P
    let Q1 := psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 1 Q
    let Q2 := psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 1 Q
    let R1 := psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 0 R -
              psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 1 R
    let R2 := psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 0 R -
              psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 1 R
    let Ga1 := psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 2 G -
               (θ1 + θ2) * psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 1 G
    let Ga2 := psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 2 G -
               (θ1 + θ2) * psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 1 G
    let Gb1 := θ1 * θ2 * psdMomentForm (momA (atoms.map realPartAtom)) (momB (atoms.map realPartAtom)) (momC (atoms.map realPartAtom)) 0 G
    let Gb2 := θ1 * θ2 * psdMomentForm (momA (atoms.map imagPartAtom)) (momB (atoms.map imagPartAtom)) (momC (atoms.map imagPartAtom)) 0 G
    change (P1 + P2) + (Q1 + Q2) + (R1 + R2) + ((Ga1 + Ga2) + (Gb1 + Gb2)) =
           (P1 + Q1 + R1 + (Ga1 + Gb1)) + (P2 + Q2 + R2 + (Ga2 + Gb2))
    rw [h4 P1 P2 Q1 Q2, h4 (P1 + Q1) (P2 + Q2) R1 R2, h4 Ga1 Ga2 Gb1 Gb2,
        h4 (P1 + Q1 + R1) (P2 + Q2 + R2) (Ga1 + Gb1) (Ga2 + Gb2)]
  have hnn := OrdField.add_le_add hre him
  rw [OrdField.zero_add, ← hsplit] at hnn
  exact OrdField.lt_irrefl (0 : K) (OrdField.lt_of_le_of_lt hnn hneg)

end ComplexExtensions

end QcQmcCert
