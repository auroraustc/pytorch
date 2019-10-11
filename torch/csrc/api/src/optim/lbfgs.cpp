#include <torch/optim/lbfgs.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>
#include <vector>


namespace torch {
namespace optim {

/**
 * Newly added.
 */
Tensor cubic_interpolate(Tensor& x1, Tensor& f1, Tensor& g1, Tensor& x2, Tensor& f2, Tensor& g2, Tensor bounds = torch::empty({0})) {
  torch::Tensor xmin_bound, xmax_bound;
  if (bounds.size(0) != 0) {
    xmin_bound = bounds[0];
    xmax_bound = bounds[1];
  } else {
    if ((x1 <= x2).item<bool>()) {
      xmin_bound = x1;
      xmax_bound = x2;
    } else {
      xmin_bound = x2;
      xmax_bound = x1;
      }
  }
  torch::Tensor d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2);
  torch::Tensor d2_square = d1 * d1 - g1 * g2;
  if ((d2_square >= 0).item<bool>()) {
    torch::Tensor d2 = d2_square.sqrt();
    torch::Tensor min_pos;
    if ((x1 <= x2).item<bool>()) {
      min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2));
    } else {
      min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2));
    }
    return torch::min(torch::max(min_pos, xmin_bound), xmax_bound);
  } else {
    return (xmin_bound + xmax_bound) / 2.;
  }
}

std::tuple<Tensor, Tensor, Tensor, int64_t> strong_wolfe(std::function<std::vector<Tensor>(std::vector<Tensor>& x, Tensor& t, Tensor& d)> obj_func, std::vector<Tensor>& x, Tensor & t, Tensor & d, Tensor & f, Tensor & g, Tensor & gtd, double c1 = 1e-4, double c2 = 0.9, double tolerance_change = 1e-9, int64_t max_ls=25) {
  // ported from torch/optim/lbfgs.py # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
  torch::Tensor d_norm = d.abs().max();
  g = g.clone();
  // evaluate objective and gradient using initial step
  std::vector<Tensor> obj_func_return = obj_func(x, t, d);
  torch::Tensor f_new = obj_func_return[0];
  torch::Tensor g_new = obj_func_return[1];
  int64_t ls_func_evals = 1;
  torch::Tensor gtd_new = g_new.dot(d);

  // bracket an interval containing a point satisfying the Wolfe criteria
  torch::Tensor t_prev = torch::zeros(1, t.options());
  torch::Tensor f_prev = f;
  torch::Tensor g_prev = g;
  torch::Tensor gtd_prev = gtd;
  bool done = false;
  int64_t ls_iter = 0;
  std::vector<Tensor> bracket;
  std::vector<Tensor> bracket_f;
  std::vector<Tensor> bracket_g;
  std::vector<Tensor> bracket_gtd;
  while (ls_iter < max_ls) {
    // check conditions
    if (((f_new > (f + c1 * t * gtd)).item<bool>()) || ((ls_iter > 1) && ((f_new >= f_prev).item<bool>()))) {
      bracket.resize(2); bracket.at(0) = t_prev; bracket.at(1) = t;
      bracket_f.resize(2); bracket_f.at(0) = f_prev; bracket_f.at(1) = f_new;
      bracket_g.resize(2); bracket_g.at(0) = g_prev; bracket_g.at(1) = g_new.clone();
      bracket_gtd.resize(2); bracket_gtd.at(0) = gtd_prev; bracket_gtd.at(1) = gtd_new;
      break;
    }
    if ((torch::abs(gtd_new) <= (-c2 * gtd)).item<bool>()) {
      bracket.resize(1); bracket.at(0) = t;
      bracket_f.resize(1); bracket_f.at(0) = f_new;
      bracket_g.resize(1); bracket_g.at(0) = g_new;
      done = true;
    }
    if ((gtd_new >= 0).item<bool>()) {
      bracket.resize(2); bracket.at(0) = t_prev; bracket.at(1) = t;
      bracket_f.resize(2); bracket_f.at(0) = f_prev; bracket_f.at(1) = f_new;
      bracket_g.resize(2); bracket_g.at(0) = g_prev; bracket_g.at(1) = g_new.clone();
      bracket_gtd.resize(2); bracket_gtd.at(0) = gtd_prev; bracket_gtd.at(1) = gtd_new;
      break;
    }
    // interpolate
    torch::Tensor bounds = torch::zeros({2,1}, t.options());
    torch::Tensor tmp = t;
    bounds[0] = t + 0.01 * (t - t_prev);
    bounds[1] = t * 10;
    t = cubic_interpolate(t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds);
    // next step
    t_prev = tmp;
    f_prev = f_new;
    g_prev = g_new.clone();
    gtd_prev = gtd_new;
    obj_func_return = obj_func(x, t, d);
    f_new = obj_func_return[0];
    g_new = obj_func_return[1];
    ls_func_evals += 1;
    gtd_new = g_new.dot(d);
    ls_iter += 1;
  }
  // reached max number of iterations?
  if (ls_iter == max_ls) {
    bracket.resize(2); bracket.at(0) = torch::zeros(1, t.options()); bracket.at(1) = t;
    bracket_f.resize(2); bracket_f.at(0) = f; bracket_f.at(1) = f_new;
    bracket_g.resize(2); bracket_g.at(0) = g; bracket_g.at(1) = g_new;
  }

  /*
   * zoom phase: we now have a point satisfying the criteria, or
   * a bracket around it. We refine the bracket until we find the
   * exact point satisfying the criteria
   */
  bool insuf_progress = false;
  // find high and low points in bracket
  int low_pos;
  int high_pos;
  if ((bracket_f.at(0) <= bracket_f.at(bracket_f.size() - 1)).item<bool>()) {
    low_pos = 0;
    high_pos = 1;
  } else {
    low_pos = 1;
    high_pos = 0;
  }
  while ((!done) && (ls_iter < max_ls)) {
    // compute new trial value
    t = cubic_interpolate(bracket.at(0), bracket_f.at(0), bracket_gtd.at(0), bracket.at(1), bracket_f.at(1), bracket_gtd.at(1));
    /**
     * test that we are making sufficient progress:
     * in case `t` is so close to boundary, we mark that we are making
     * insufficient progress, and if
     *   + we have made insufficient progress in the last step, or
     *   + `t` is at one of the boundary,
     * we will move `t` to a position which is `0.1 * len(bracket)`
     * away from the nearest boundary point.
     */
    torch::Tensor max_bracket = torch::max(bracket.at(0), bracket.at(bracket.size() - 1));
    torch::Tensor min_bracket = torch::min(bracket.at(0), bracket.at(bracket.size() - 1));
    torch::Tensor eps = 0.1 * (max_bracket - min_bracket);
    if ((torch::min(max_bracket - t, t - min_bracket) < eps).item<bool>()) {
      // interpolation close to boundary
      if ((insuf_progress) || (t >= max_bracket).item<bool>() || (t <= min_bracket).item<bool>()) {
        // evaluate at 0.1 away from boundary
        if ((torch::abs(t - max_bracket) < torch::abs(t - min_bracket)).item<bool>()) {
          t = max_bracket - eps;
        } else {
          t = min_bracket + eps;
        }
        insuf_progress = false;
      } else {
        insuf_progress = true;
      }
    } else {
      insuf_progress = false;
    }

    // Evaluate new point
    obj_func_return = obj_func(x, t, d);
    f_new = obj_func_return[0];
    g_new = obj_func_return[1];
    ls_func_evals += 1;
    gtd_new = g_new.dot(d);
    ls_iter += 1;
    if ((f_new > (f + c1 * t * gtd)).item<bool>() || (f_new >= bracket_f.at(low_pos)).item<bool>()) {
      // Armijo condition not satisfied or not lower than lowest point
      bracket.at(high_pos) = t;
      bracket_f.at(high_pos) = f_new;
      bracket_g.at(high_pos) = g_new.clone();
      bracket_gtd.at(high_pos) = gtd_new;
      if ((bracket_f.at(0) <= bracket_f.at(1)).item<bool>()) {
        low_pos = 0;
        high_pos = 1;
      } else {
        low_pos = 1;
        high_pos = 0;
      }
    } else {
      if ((torch::abs(gtd_new) <= (-c2 * gtd)).item<bool>()) {
        // Wolfe conditions satisfied
        done = true;
      } else if ((gtd_new * (bracket.at(high_pos) - bracket.at(low_pos)) >= 0).item<bool>()) {
        // old high becomes new low
        bracket.at(high_pos) = bracket.at(low_pos);
        bracket_f.at(high_pos) = bracket_f.at(low_pos);
        bracket_g.at(high_pos) = bracket_g.at(low_pos);
        bracket_gtd.at(high_pos) = bracket_gtd.at(low_pos);
      }

      // new point becomes new low
      bracket.at(low_pos) = t;
      bracket_f.at(low_pos) = f_new;
      bracket_g.at(low_pos) = g_new.clone();
      bracket_gtd.at(low_pos) = gtd_new;
    }

    // line-search bracket is so small
    if ((abs(bracket.at(1) - bracket.at(0)) * d_norm < tolerance_change).item<bool>()) {
      break;
    }
  } // end while

  t = bracket.at(low_pos);
  f_new = bracket_f.at(low_pos);
  g_new = bracket_g.at(low_pos);

  return std::make_tuple(f_new, g_new, t, ls_func_evals);
}

LBFGSOptions::LBFGSOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

Tensor LBFGS::gather_flat_grad() {
  std::vector<Tensor> views;
  for (auto& parameter : parameters_) {
    if (!parameter.grad().defined()) {
      views.push_back(parameter.new_empty({parameter.numel()}).zero_());
    }
    else if (parameter.grad().is_sparse()) {
      views.push_back(parameter.grad().to_dense().view(-1));
    }
    else {
      views.push_back(parameter.grad().view(-1));
    }
  }
  return torch::cat(views);
}

void LBFGS::add_grad(const torch::Tensor& step_size, const Tensor& update) {
  NoGradGuard guard;
  int64_t offset = 0;
  for (auto& parameter : parameters_) {
    int64_t numel = parameter.numel();
    parameter.add_(
        update.slice(0, offset, offset + numel, 1).view_as(parameter),
        step_size.item<float>());
    offset += numel;
  }
}

/**
 * Newly added.
 */
std::vector<Tensor> LBFGS::clone_param() {
  std::vector<Tensor> p;
  for (auto& parameter : parameters_) {
    p.push_back(parameter.clone());
  }
  return p;
}

void LBFGS::set_param(std::vector<Tensor>& parameters_data) {
  NoGradGuard guard;
  for (int i = 0; i <= parameters_.size() - 1; i++) {
    parameters_.at(i).copy_(parameters_data.at(i));
  }
  return;
}

std::vector<Tensor> LBFGS::directional_evaluate(LossClosure closure, std::vector<Tensor>& x, Tensor& t, Tensor& d) {
  add_grad(t, d);
  torch::Tensor loss = closure().clone().detach();
  torch::Tensor flat_grad = gather_flat_grad();
  set_param(x);
  return {loss, flat_grad};
}

torch::Tensor LBFGS::step(LossClosure closure) {
  torch::Tensor orig_loss = closure();
  torch::Tensor loss = orig_loss.clone().detach();
  int64_t current_evals = 1;
  func_evals += 1;

  Tensor flat_grad = gather_flat_grad();
  Tensor abs_grad_sum = flat_grad.abs().sum();

  if (abs_grad_sum.item<float>() <= options.tolerance_grad()) {
    return loss;
  }

  Tensor ONE = torch::tensor(1, flat_grad.options());

  int64_t n_iter = 0;
  while (n_iter < options.max_iter()) {
    n_iter++;
    state_n_iter++;

    if (state_n_iter == 1) {
      d = flat_grad.neg();
      H_diag = ONE;
      prev_flat_grad = flat_grad.clone();
    } else {
      Tensor y = flat_grad.sub(prev_flat_grad);
      Tensor s = d.mul(t);
      Tensor ys = y.dot(s);

      if (ys.item<float>() > 1e-10) {
        // updating memory

        if (old_dirs.size() == options.history_size()) {
          // shift history by one (limited memory)
          old_dirs.pop_front();
          old_stps.pop_front();
        }

        // store new direction/step
        old_dirs.push_back(y);
        old_stps.push_back(s);

        // update scale of initial Hessian approximation
        H_diag = ys / y.dot(y);
      }

      int64_t num_old = old_dirs.size();

      for (int64_t i = 0; i < num_old; i++) {
        ro.at(i) = ONE / old_dirs.at(i).dot(old_stps.at(i));
      }

      Tensor q = flat_grad.neg();
      for (int64_t i = num_old - 1; i >= 0; i--) {
        al.at(i) = old_stps.at(i).dot(q) * ro.at(i);
        q.add_(old_dirs.at(i), -al.at(i).item());
      }

      // Multiply by initial Hessian
      // r/d is the final direction
      Tensor r = q.mul(H_diag);
      d = r;

      for (int64_t i = 0; i < num_old; i++) {
        Tensor be_i = old_dirs.at(i).dot(r) * ro.at(i);
        r.add_(old_stps.at(i), (al.at(i) - be_i).item());
      }
      prev_flat_grad.copy_(flat_grad);
    }
    // prev_loss = loss;
    /**
     * compute step length
     */

    // reset initial guess for step size
    if (n_iter == 1) {
      t = torch::min(ONE, ONE / abs_grad_sum) * options.learning_rate();
    } else {
      t = torch::tensor(options.learning_rate(), flat_grad.options());
    }

    Tensor gtd = flat_grad.dot(d);
    // std::cout << d.sizes() << std::endl;
    int64_t ls_func_evals = 0;
    if (options.line_search_fn() != 0) {
      /**
        * perform line search, using user function
        * 1: strong_wolfe
        */
      if (options.line_search_fn() == 1) {
        std::vector<Tensor> x_init = clone_param();
        std::function<std::vector<Tensor>(std::vector<Tensor>& x, Tensor& t, Tensor& d)> obj_func = [&](std::vector<Tensor>& x, Tensor& t, Tensor& d){
          return directional_evaluate(closure, x, t, d);
        };
        std::tie(loss, flat_grad, t, ls_func_evals) = strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd);
        add_grad(t, d);
        abs_grad_sum = flat_grad.abs().sum();
      }
    } else {
      add_grad(t, d);
      if (n_iter != options.max_iter()) {
        // re-evaluate function only if not in last iteration
        // the reason we do this: in a stochastic setting,
        // no use to re-evaluate that function here
        loss = closure();
        flat_grad = gather_flat_grad();
        abs_grad_sum = flat_grad.abs().sum();
        ls_func_evals = 1;
      }
    }
    

    current_evals += ls_func_evals;
    // func_evals += ls_func_evals;

    /**
     * Check conditions
     */

    if (n_iter == options.max_iter()) {
      break;
    } else if (current_evals >= options.max_eval()) {
      break;
    } else if (abs_grad_sum.item<float>() <= options.tolerance_grad()) {
      break;
    } else if (gtd.item<float>() > -options.tolerance_grad()) {
      break;
    } else if (
        d.mul(t).abs_().sum().item<float>() <= options.tolerance_change()) {
      break;
    } else if (
        std::abs(loss.item<float>() - prev_loss.item<float>()) <
        options.tolerance_change()) {
      break;
    }
  }
  return orig_loss;
}

void LBFGS::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void LBFGS::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
