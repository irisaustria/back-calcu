from flask import Flask, request, jsonify
from flask_cors import CORS
import sympy as sp
from sympy import Symbol
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

app = Flask(__name__)
CORS(app)

x = Symbol('x', real=True)
y = Symbol('y', real=True)
t = Symbol('t', positive=True, real=True)
s = Symbol('s', real=True)
C1 = Symbol('C')

transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)


def parse_math(expr_str):
    local_dict = {
        'x': x,
        'y': y,
        't': t,
        's': s,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'sqrt': sp.sqrt,
        'log': sp.log,
        'ln': sp.log,
        'pi': sp.pi,
        'E': sp.E,
        'Heaviside': sp.Heaviside
    }
    return parse_expr(
        expr_str,
        local_dict=local_dict,
        transformations=transformations,
        evaluate=False
    )


def pretty_latex(expr):
    txt = sp.latex(expr)
    txt = txt.replace(r"\operatorname{atan}", r"\arctan")
    txt = txt.replace(r"\operatorname{asin}", r"\arcsin")
    txt = txt.replace(r"\operatorname{acos}", r"\arccos")
    return txt


def json_ok(title, input_latex, result_latex, steps):
    return jsonify({
        "title": title,
        "input_latex": input_latex,
        "result_latex": result_latex,
        "steps": steps
    })


# -------------------------
# SEPARABLES
# -------------------------
def solve_separable(fx_str, gy_str):
    if not fx_str or not gy_str:
        raise ValueError("Debes capturar f(x) y g(y).")

    fx = parse_math(fx_str)
    gy = parse_math(gy_str)

    original_rhs = sp.simplify(fx * gy)
    left_integral = sp.simplify(sp.integrate(1 / gy, y))
    right_integral = sp.simplify(sp.integrate(fx, x))

    result_lhs = left_integral
    result_rhs = right_integral + C1

    steps = [
        "Partimos de la ecuación diferencial:",
        f"$\\dfrac{{dy}}{{dx}} = {pretty_latex(original_rhs)}$.",
        "Separamos variables:",
        f"$\\dfrac{{1}}{{{pretty_latex(gy)}}}\\,dy = {pretty_latex(fx)}\\,dx$.",
        "Integramos ambos lados:",
        f"$\\int \\dfrac{{1}}{{{pretty_latex(gy)}}}\\,dy = \\int {pretty_latex(fx)}\\,dx$.",
        "Evaluamos las integrales:",
        f"Lado izquierdo: ${pretty_latex(left_integral)}$.",
        f"Lado derecho: ${pretty_latex(right_integral)}$.",
        "Por tanto, la solución implícita es:",
        f"${pretty_latex(result_lhs)} = {pretty_latex(result_rhs)}$."
    ]

    input_latex = r"\frac{dy}{dx} = " + pretty_latex(original_rhs)
    result_latex = pretty_latex(result_lhs) + " = " + pretty_latex(result_rhs)

    return "Separación de Variables", input_latex, result_latex, steps


# -------------------------
# EXACTAS
# -------------------------
def solve_exact(M_str, N_str):
    if not M_str or not N_str:
        raise ValueError("Debes capturar M(x,y) y N(x,y).")

    M = parse_math(M_str)
    N = parse_math(N_str)

    My = sp.simplify(sp.diff(M, y))
    Nx = sp.simplify(sp.diff(N, x))

    if sp.simplify(My - Nx) != 0:
        raise ValueError("La ecuación no es exacta, porque ∂M/∂y y ∂N/∂x no coinciden.")

    psi_partial = sp.integrate(M, x)
    psi_y = sp.diff(psi_partial, y)
    h_prime = sp.simplify(N - psi_y)
    h = sp.integrate(h_prime, y)
    psi_total = sp.simplify(psi_partial + h)

    result_lhs = psi_total
    result_rhs = C1

    steps = [
        "Identificamos las funciones de la ecuación diferencial:",
        f"$M(x,y) = {pretty_latex(M)}$.",
        f"$N(x,y) = {pretty_latex(N)}$.",
        "Calculamos las derivadas parciales:",
        f"$\\dfrac{{\\partial M}}{{\\partial y}} = {pretty_latex(My)}$.",
        f"$\\dfrac{{\\partial N}}{{\\partial x}} = {pretty_latex(Nx)}$.",
        "Como ambas derivadas parciales son iguales, la ecuación es exacta.",
        "Buscamos una función potencial \\(\\psi(x,y)\\) tal que:",
        f"$\\psi_x = {pretty_latex(M)}$ y $\\psi_y = {pretty_latex(N)}$.",
        "Integramos \\(M(x,y)\\) respecto de \\(x\\):",
        f"$\\psi(x,y) = \\int {pretty_latex(M)}\\,dx = {pretty_latex(psi_partial)} + h(y)$.",
        "Derivamos respecto de \\(y\\) y comparamos con \\(N(x,y)\\):",
        f"$\\psi_y = {pretty_latex(psi_y)} + h'(y)$.",
        f"De aquí, $h'(y) = {pretty_latex(h_prime)}$.",
        "Integramos la parte restante:",
        f"$h(y) = {pretty_latex(h)}$.",
        "Finalmente, la solución implícita es:",
        f"${pretty_latex(result_lhs)} = {pretty_latex(result_rhs)}$."
    ]

    input_latex = pretty_latex(M) + r"\,dx + " + pretty_latex(N) + r"\,dy = 0"
    result_latex = pretty_latex(result_lhs) + " = " + pretty_latex(result_rhs)

    return "Ecuación Exacta", input_latex, result_latex, steps


# -------------------------
# FACTOR INTEGRANTE
# -------------------------
def solve_integrating(M_str, N_str):
    if not M_str or not N_str:
        raise ValueError("Debes capturar M(x,y) y N(x,y).")

    M = parse_math(M_str)
    N = parse_math(N_str)

    My = sp.simplify(sp.diff(M, y))
    Nx = sp.simplify(sp.diff(N, x))

    steps = [
        "Identificamos las funciones de la ecuación diferencial:",
        f"$M(x,y) = {pretty_latex(M)}$.",
        f"$N(x,y) = {pretty_latex(N)}$.",
        "Calculamos las derivadas parciales:",
        f"$\\dfrac{{\\partial M}}{{\\partial y}} = {pretty_latex(My)}$.",
        f"$\\dfrac{{\\partial N}}{{\\partial x}} = {pretty_latex(Nx)}$."
    ]

    if sp.simplify(My - Nx) == 0:
        psi_partial = sp.integrate(M, x)
        psi_y = sp.diff(psi_partial, y)
        h_prime = sp.simplify(N - psi_y)
        h = sp.integrate(h_prime, y)
        psi_total = sp.simplify(psi_partial + h)

        steps.extend([
            "La ecuación ya es exacta, por lo que no requiere factor integrante.",
            f"Por tanto, la solución implícita es: ${pretty_latex(psi_total)} = {pretty_latex(C1)}$."
        ])

        input_latex = pretty_latex(M) + r"\,dx + " + pretty_latex(N) + r"\,dy = 0"
        result_latex = pretty_latex(psi_total) + " = " + pretty_latex(C1)
        return "Factor Integrante", input_latex, result_latex, steps

    mu = None

    if sp.simplify(N) != 0:
        rx = sp.simplify((My - Nx) / N)
        if not rx.has(y):
            mu = sp.simplify(sp.exp(sp.integrate(rx, x)))
            steps.extend([
                "Buscamos un factor integrante dependiente solo de \\(x\\):",
                f"$\\dfrac{{M_y - N_x}}{{N}} = {pretty_latex(rx)}$.",
                "Como esta expresión depende únicamente de \\(x\\), existe \\(\\mu(x)\\).",
                f"$\\mu(x) = e^{{\\int {pretty_latex(rx)}\\,dx}} = {pretty_latex(mu)}$."
            ])

    if mu is None and sp.simplify(M) != 0:
        ry = sp.simplify((Nx - My) / M)
        if not ry.has(x):
            mu = sp.simplify(sp.exp(sp.integrate(ry, y)))
            steps.extend([
                "Buscamos un factor integrante dependiente solo de \\(y\\):",
                f"$\\dfrac{{N_x - M_y}}{{M}} = {pretty_latex(ry)}$.",
                "Como esta expresión depende únicamente de \\(y\\), existe \\(\\mu(y)\\).",
                f"$\\mu(y) = e^{{\\int {pretty_latex(ry)}\\,dy}} = {pretty_latex(mu)}$."
            ])

    if mu is None:
        raise ValueError("No se encontró un factor integrante simple dependiente solo de x o solo de y.")

    M2 = sp.simplify(mu * M)
    N2 = sp.simplify(mu * N)

    if sp.simplify(sp.diff(M2, y) - sp.diff(N2, x)) != 0:
        raise ValueError("El factor integrante encontrado no convierte la ecuación en exacta.")

    steps.extend([
        "Multiplicamos toda la ecuación por el factor integrante:",
        f"$\\widetilde{{M}} = {pretty_latex(M2)}$.",
        f"$\\widetilde{{N}} = {pretty_latex(N2)}$.",
        "La ecuación resultante es exacta."
    ])

    psi_partial = sp.integrate(M2, x)
    psi_y = sp.diff(psi_partial, y)
    h_prime = sp.simplify(N2 - psi_y)
    h = sp.integrate(h_prime, y)
    psi_total = sp.simplify(psi_partial + h)

    steps.extend([
        "Integramos \\(\\widetilde{M}\\) respecto de \\(x\\):",
        f"$\\psi(x,y) = {pretty_latex(psi_partial)} + h(y)$.",
        "Derivamos respecto de \\(y\\) y comparamos con \\(\\widetilde{N}\\):",
        f"$\\psi_y = {pretty_latex(psi_y)} + h'(y)$.",
        f"De aquí, $h'(y) = {pretty_latex(h_prime)}$.",
        "Integramos la parte restante:",
        f"$h(y) = {pretty_latex(h)}$.",
        "Así, la solución implícita queda dada por:",
        f"${pretty_latex(psi_total)} = {pretty_latex(C1)}$."
    ])

    input_latex = pretty_latex(M) + r"\,dx + " + pretty_latex(N) + r"\,dy = 0"
    result_latex = pretty_latex(psi_total) + " = " + pretty_latex(C1)

    return "Factor Integrante", input_latex, result_latex, steps


# -------------------------
# LINEALES
# -------------------------
def solve_linear(P_str, Q_str):
    if not P_str or not Q_str:
        raise ValueError("Debes capturar P(x) y Q(x).")

    P = parse_math(P_str)
    Q = parse_math(Q_str)

    mu = sp.simplify(sp.exp(sp.integrate(P, x)))
    rhs = sp.simplify(mu * Q)
    rhs_int = sp.integrate(rhs, x)
    solution_rhs = sp.simplify((rhs_int + C1) / mu)

    steps = [
        "Partimos de la ecuación diferencial lineal:",
        f"$\\dfrac{{dy}}{{dx}} + ({pretty_latex(P)})y = {pretty_latex(Q)}$.",
        "Identificamos las funciones correspondientes:",
        f"$P(x) = {pretty_latex(P)}$.",
        f"$Q(x) = {pretty_latex(Q)}$.",
        "Calculamos el factor integrante:",
        f"$\\mu(x) = e^{{\\int P(x)\\,dx}} = {pretty_latex(mu)}$.",
        "Multiplicamos toda la ecuación por el factor integrante:",
        f"$\\dfrac{{d}}{{dx}}\\left({pretty_latex(mu)}y\\right) = {pretty_latex(rhs)}$.",
        "Integramos ambos lados:",
        f"${pretty_latex(mu)}y = {pretty_latex(rhs_int)} + {pretty_latex(C1)}$.",
        "Despejamos la función \\(y\\):",
        f"$y = {pretty_latex(solution_rhs)}$."
    ]

    input_latex = r"\frac{dy}{dx} + (" + pretty_latex(P) + r")y = " + pretty_latex(Q)
    result_latex = "y = " + pretty_latex(solution_rhs)

    return "Ecuación Lineal", input_latex, result_latex, steps


# -------------------------
# LAPLACE CON REGLAS EXPLÍCITAS
# -------------------------
def factor_exp_shift(expr):
    """
    Detecta expresiones de la forma exp(a*t)*g(t)
    y devuelve (a, g). Si no, devuelve (None, expr).
    """
    if expr.func == sp.exp:
        expo = expr.args[0]
        if expo.has(t):
            a = sp.simplify(expo / t)
            if expo == a * t and a.is_number:
                return a, sp.Integer(1)
        return None, expr

    if expr.is_Mul:
        exp_factor = None
        others = []
        for arg in expr.args:
            if arg.func == sp.exp:
                expo = arg.args[0]
                if expo.has(t):
                    a = sp.simplify(expo / t)
                    if expo == a * t and a.is_number:
                        exp_factor = a
                    else:
                        others.append(arg)
                else:
                    others.append(arg)
            else:
                others.append(arg)

        if exp_factor is not None:
            g = sp.Mul(*others) if others else sp.Integer(1)
            return exp_factor, g

    return None, expr


def laplace_term_steps(expr):
    steps = []

    coeff, core = expr.as_coeff_Mul()

    if coeff != 1:
        steps.append(
            f"Aplicamos linealidad y sacamos la constante:"
        )
        steps.append(
            f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(coeff)}\\,\\mathcal{{L}}\\{{{pretty_latex(core)}\\}}$."
        )
        inner_steps, inner_result = laplace_term_steps(core)
        steps.extend(inner_steps)
        result = sp.simplify(coeff * inner_result)
        steps.append(
            f"Multiplicando por la constante, obtenemos:"
        )
        steps.append(
            f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(result)}$."
        )
        return steps, result

    a, g = factor_exp_shift(expr)
    if a is not None and g != expr:
        steps.append(
            "Usamos la propiedad de corrimiento exponencial:"
        )
        steps.append(
            r"$\mathcal{L}\{e^{at}f(t)\}=F(s-a)$."
        )
        steps.append(
            f"Aquí, $a={pretty_latex(a)}$ y la función base es $f(t)={pretty_latex(g)}$."
        )
        inner_steps, inner_result = laplace_term_steps(g)
        steps.extend(inner_steps)
        shifted = sp.simplify(inner_result.subs(s, s - a))
        steps.append(
            f"Sustituimos $s \\mapsto s-{pretty_latex(a)}$ en la transformada obtenida."
        )
        steps.append(
            f"Por tanto,"
        )
        steps.append(
            f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(shifted)}$."
        )
        return steps, shifted

    if not expr.has(t):
        result = sp.simplify(expr / s)
        steps.append("Aplicamos la transformada de una constante:")
        steps.append(
            f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = \\dfrac{{{pretty_latex(expr)}}}{{s}}$."
        )
        return steps, result

    if expr == t:
        result = 1 / s**2
        steps.append("Usamos la fórmula conocida:")
        steps.append(r"$\mathcal{L}\{t\}=\dfrac{1}{s^2}$.")
        return steps, result

    if expr.is_Pow and expr.base == t and expr.exp.is_integer and expr.exp >= 0:
        n = int(expr.exp)
        result = sp.factorial(n) / s**(n + 1)
        steps.append("Usamos la fórmula de la transformada de una potencia:")
        steps.append(
            f"$\\mathcal{{L}}\\{{t^{n}\\}} = \\dfrac{{n!}}{{s^{{n+1}}}}$, con $n={n}$."
        )
        steps.append(
            f"Entonces,"
        )
        steps.append(
            f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(result)}$."
        )
        return steps, result

    if expr.func == sp.sin:
        arg = expr.args[0]
        b = sp.simplify(arg / t)
        if arg == b * t and b.is_number:
            result = b / (s**2 + b**2)
            steps.append("Usamos la fórmula de la transformada del seno:")
            steps.append(
                r"$\mathcal{L}\{\sin(bt)\}=\dfrac{b}{s^2+b^2}$."
            )
            steps.append(
                f"Aquí, $b={pretty_latex(b)}$."
            )
            steps.append(
                f"Por tanto,"
            )
            steps.append(
                f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(result)}$."
            )
            return steps, result

    if expr.func == sp.cos:
        arg = expr.args[0]
        b = sp.simplify(arg / t)
        if arg == b * t and b.is_number:
            result = s / (s**2 + b**2)
            steps.append("Usamos la fórmula de la transformada del coseno:")
            steps.append(
                r"$\mathcal{L}\{\cos(bt)\}=\dfrac{s}{s^2+b^2}$."
            )
            steps.append(
                f"Aquí, $b={pretty_latex(b)}$."
            )
            steps.append(
                f"Por tanto,"
            )
            steps.append(
                f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(result)}$."
            )
            return steps, result

    if expr.func == sp.exp:
        expo = expr.args[0]
        a = sp.simplify(expo / t)
        if expo == a * t and a.is_number:
            result = 1 / (s - a)
            steps.append("Usamos la fórmula de la transformada exponencial:")
            steps.append(
                r"$\mathcal{L}\{e^{at}\}=\dfrac{1}{s-a}$."
            )
            steps.append(
                f"Aquí, $a={pretty_latex(a)}$."
            )
            steps.append(
                f"Entonces,"
            )
            steps.append(
                f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(result)}$."
            )
            return steps, result

    result = sp.simplify(sp.laplace_transform(expr, t, s, noconds=True))
    steps.append("En este caso aplicamos la transformada de Laplace de forma simbólica.")
    steps.append(
        f"$\\mathcal{{L}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(result)}$."
    )
    return steps, result


def solve_laplace(expr_str, mode):
    if not expr_str:
        raise ValueError("Debes capturar una expresión.")

    expr = parse_math(expr_str)

    if mode == "forward":
        expanded = sp.expand(expr)

        steps = [
            "Identificamos la función en el dominio del tiempo:",
            f"$f(t) = {pretty_latex(expanded)}$."
        ]

        if expanded.is_Add:
            steps.append("Aplicamos la linealidad de la transformada de Laplace:")
            steps.append(
                r"$\mathcal{L}\{f_1(t)+f_2(t)\}=\mathcal{L}\{f_1(t)\}+\mathcal{L}\{f_2(t)\}$."
            )
            total = 0
            for i, term in enumerate(expanded.args, start=1):
                steps.append(f"**Término {i}.**")
                sub_steps, sub_result = laplace_term_steps(term)
                steps.extend(sub_steps)
                total += sub_result
            total = sp.simplify(total)
            steps.append("Sumamos las transformadas obtenidas:")
            steps.append(
                f"$\\mathcal{{L}}\\{{{pretty_latex(expanded)}\\}} = {pretty_latex(total)}$."
            )
            return "Laplace directa", pretty_latex(expanded), pretty_latex(total), steps

        sub_steps, result = laplace_term_steps(expanded)
        steps.extend(sub_steps)
        return "Laplace directa", pretty_latex(expanded), pretty_latex(result), steps

    if mode == "inverse":
        result = sp.simplify(sp.inverse_laplace_transform(expr, s, t))

        steps = [
            "Identificamos la función en el dominio de Laplace:",
            f"$F(s) = {pretty_latex(expr)}$.",
            "Aplicamos la transformada inversa de Laplace:",
            f"$\\mathcal{{L}}^{{-1}}\\{{{pretty_latex(expr)}\\}}$.",
            "Finalmente, obtenemos la función en el dominio del tiempo:",
            f"$\\mathcal{{L}}^{{-1}}\\{{{pretty_latex(expr)}\\}} = {pretty_latex(result)}$."
        ]

        return "Laplace inversa", pretty_latex(expr), pretty_latex(result), steps

    raise ValueError("Modo de Laplace no válido.")


@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json or {}
        method = data.get("method", "").strip()

        if method == "sep":
            title, input_latex, result_latex, steps = solve_separable(
                data.get("fx", "").strip(),
                data.get("gy", "").strip()
            )
        elif method == "exa":
            title, input_latex, result_latex, steps = solve_exact(
                data.get("M", "").strip(),
                data.get("N", "").strip()
            )
        elif method == "int":
            title, input_latex, result_latex, steps = solve_integrating(
                data.get("M", "").strip(),
                data.get("N", "").strip()
            )
        elif method == "lin":
            title, input_latex, result_latex, steps = solve_linear(
                data.get("P", "").strip(),
                data.get("Q", "").strip()
            )
        elif method == "lap":
            title, input_latex, result_latex, steps = solve_laplace(
                data.get("expression", "").strip(),
                data.get("mode", "forward").strip()
            )
        else:
            return jsonify({"error": "Método no válido."}), 400

        return json_ok(title, input_latex, result_latex, steps)

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)