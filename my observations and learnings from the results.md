# SGD Observations — Learning by Breaking Things

---

## Function 1: f(x, y) = x² + y²



| Experiment | Parameters | What I was thinking when I set this up | What I saw | What I think is going on |
|---|---|---|---|---|
| Baseline run | lr=0.1, start=(-8, 8) | Just wanted to see it work once before changing anything | It went to the origin in maybe 10 steps. Not a curve, not a spiral — just a diagonal. Loss fell from around 82 to basically 0. I honestly expected it to take longer. | I think because the contours are circles, the gradient at every point is already pointing exactly toward the center. So there's no reason to deviate. The path being a diagonal makes sense once you realize the descent direction never changes relative to the target. |
| What if I make lr very small | lr=0.01, start=(-8, 8) | I wanted to see if small lr just slows things down proportionally or if something else changes | Still heading the same way but after 100 iterations the loss was still sitting around 2. It barely got there. | So it's not just proportionally slower — you actually run out of iterations before converging. 100 steps at 0.01 is like 10 steps at 0.1. I should have thought about that before running it. The direction was always right, just the step size was too small to get anywhere in the budget. |
| Start from further away | lr=0.1, start=(15, -15) | I was curious if starting far from the minimum causes any trouble — maybe the gradient is too large and overshoots? | Same behavior as the baseline, just starting from further out. Reached the origin fine. | The gradient at (15,-15) is [30,-30] — bigger than before, so the step is bigger too. But it still points toward the minimum. The function is quadratic so the gradient scales linearly, and the step happens to stay stable. I was expecting some overshoot but there was none, which surprised me a little. |
| Push lr past 1 | lr=1.2, start=(-8, 8) | I picked this after noticing the baseline was lr=0.1 and wanted to see how high I could go before it breaks | It blew up. The path bounced around, loss went up instead of down, the whole thing diverged. | So there is a limit. Working out the update: x becomes x - lr\*2x = x(1 - 2\*lr). For this to shrink toward zero you need \|1 - 2\*lr\| < 1, which means lr has to be less than 1. At lr=1.2 the factor is -1.4, so every step flips the sign and makes it larger. I didn't guess 1.0 as the threshold  |
| Start exactly on the x-axis | lr=0.1, start=(8, 0) | Curious what happens when one coordinate is already at the minimum | x converged normally. y stayed at zero the whole time. | The gradient in y is 2y, which is 0 when y=0. So y never gets updated at all. It's just doing 1D descent on x² and y is sitting there doing nothing.  |

**What I took away from f1:** The threshold lr=1.0 isn't something I guessed, it falls out of the algebra. Also the path being a diagonal isn't obvious until you think about why the gradient always points the same direction relative to the minimum on a sphere.

---

## Function 2: f(x, y) = x² + 10y²


| Experiment | Parameters | What I was thinking when I set this up | What I saw | What I think is going on |
|---|---|---|---|---|
| Baseline with moderate lr | lr=0.05, start=(-8, 8) | Kept lr lower than f1 since I suspected the steep y direction would cause problems at 0.1 | The path made a sharp right angle. It dropped almost straight down in y first, then crept along the x-axis for the rest of the run. The loss curve looked fine but the path looked strange to me. | At the starting point, grad_y = 20\*8 = 160 and grad_x = 2\*(-8) = -16. The y component is literally 10x bigger so the first steps are almost entirely in y. Once y is near zero, the y gradient vanishes and only the x gradient does anything. The right angle in the path is just the algorithm handling one dimension at a time because the gradients are so unbalanced. |
| Lower lr | lr=0.01, start=(-8, 8) | Wondering if a smaller lr fixes the right-angle shape or if it's just a property of the function | Same right-angle shape. Slower, but structurally identical. | The shape comes from the curvature mismatch, not from the learning rate. Changing lr just changes how big each step is — it doesn't fix the fact that grad_y >> grad_x at this starting point. The right-angle path is a feature of this landscape regardless of how carefully you step. |
| Push lr too high | lr=0.2, start=(-8, 8) | I wanted to find where it breaks like I did with f1. Based on the math the threshold should be lower here. | Diverged. The y direction exploded. | The update in y is y(1 - 20\*lr). For stability: \|1 - 20\*lr\| < 1, so lr < 0.1. At 0.2, the factor is 1 - 4 = -3. Every y step triples in magnitude and flips sign. x was still stable (since lr < 1 for x) but y going haywire is enough to break everything. One unstable direction is all it takes. |
| Start near the x-axis | lr=0.05, start=(8, 0.5) | If y starts small, is the problem less severe? | Much cleaner. It still showed the right-angle shape faintly but converged much faster overall. | When y is small, grad_y = 20\*0.5 = 10, which is now comparable to grad_x = 16. The imbalance is much less severe.  |
| Start from far away | lr=0.05, start=(20, -20) | Just checking if the same pattern holds at larger scale | Same right-angle path, just bigger. Qualitatively identical. | The curvature mismatch is a property of the function geometry, not of how far you start.. |

**What I took away from f2:** The maximum safe learning rate is set by the steepest curvature, not the average. The Hessian here is diag(2, 20) and the threshold comes from the eigenvalue 20, not 2. So you're forced to use a small lr everywhere just because of one bad direction. This is why Adam and RMSProp exist — they scale the step per dimension so you don't have to choose between being slow in x and exploding in y.

---

## Function 3: f(x, y) = x² - y²
**This one has a saddle point at the origin. 


| Experiment | Parameters | What I was thinking when I set this up | What I saw | What I think is going on |
|---|---|---|---|---|
| Start away from saddle | lr=0.05, start=(2, 2) | Genuinely didn't know what would happen. Maybe it oscillates around the saddle? | x went to zero quickly. y shot up to around 27000. It's "converging" but in the wrong direction entirely. | The update in y is y - lr\*(-2y) = y(1 + 2\*lr) = y\*1.1.  And since the function is x²-y², a large y gives a very negative loss. The optimizer is technically descending but the function has no floor in the y direction — it just keeps going. This isn't the algorithm working. The objective is unbounded. |
| Start very close to saddle | lr=0.05, start=(0.1, 0.1) | If I start near the saddle, does it stay near it? | No. Same thing happens, just takes longer to get going since y=0.1 is small initially. Eventually it escapes and diverges. | y(1.1)^n diverges for any nonzero y. Even y=0.1 will eventually escape. The saddle point is unstable in the y direction  |
| Start exactly at the saddle | lr=0.05, start=(0, 0) | If the saddle is at the origin, and I start there, does it just stay? | Yes. Nothing moved. Gradient is [0,0] at the origin so the algorithm had nothing to follow. | This is actually a problem, not a success. The algorithm stopped at a point that is not a minimum — it's a saddle. From outside it looks like convergence. But the function value at (0,0) is 0, and any point with y > 0 gives a negative value. Gradient descent can't tell the difference between this and an actual minimum, which is unsettling. |
| Start on the y-axis | lr=0.05, start=(0, 2) | x=0 means grad_x = 0, so x should never move. y should diverge on its own. | x stayed at 0, y shot upward on its own. | grad_x = 2\*0 = 0 always since x never changes. I set this up specifically to isolate the y behavior without x involved, and it confirmed what the math said. |
| Start on the x-axis | lr=0.05, start=(2, 0) | y=0 means grad_y = 0, so y should never move. x should converge. | x converged to 0. y stayed at 0 the whole time. | grad_y = -2\*0 = 0 always. So y is frozen and x just does its normal thing. The x-axis is technically a stable manifold of the saddle — if you land on it exactly, you stay on it.  |



---


The learning rate question I thought was just a tuning thing turned out to have an exact answer derivable from the function. For f1 the limit is lr < 1. For f2 it's lr < 0.1.  The steepest direction sets the limit for the whole problem.   all dimensions obviously do not converge at the same speed even with the same lr.

One thing I'm still not fully satisfied with: in f3, starting exactly at (0,0) stops the algorithm. But in a neural network you'd never initialize everything to zero exactly. So in practice the saddle point problem isn't "gets stuck at the saddle" but rather "starts near a saddle and either escapes slowly or diverges depending on direction." That seems harder to reason about and these experiments didn't fully capture that.
