from flask import Flask, request, jsonify, render_template
import itertools
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)


class HashTable:
    def __init__(self):
        self.data = {}

    def put(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)


lines = HashTable()
lines.put("A1", 35)
lines.put("A2", 32)
lines.put("B1", 38)
lines.put("B2", 41)


genes = HashTable()
genes.put("A1", "Rr")
genes.put("A2", "rr")
genes.put("B1", "RR")
genes.put("B2", "Rr")


def normalize_pair(pair):
    a, b = pair[0], pair[1]
    if a.islower() and b.isupper():
        return b + a
    return a + b


def punnett_square(genotype_a, genotype_b):
    offspring = []
    for x in genotype_a:
        for y in genotype_b:
            offspring.append(normalize_pair(x + y))

    counts = {}
    for g in offspring:
        counts[g] = counts.get(g, 0) + 1
    total = len(offspring)

    genotype_percent = {g: round(c / total * 100, 1) for g, c in counts.items()}
    dominant_count = sum(c for g, c in counts.items() if g[0].isupper())

    return {
        "genotype_percent": genotype_percent,
        "resistant_percent": round(dominant_count / total * 100, 1),
        "susceptible_percent": round((total - dominant_count) / total * 100, 1),
    }


class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def tree_to_dict(node):
    if node is None:
        return None
    return {
        "name": node.name,
        "left": tree_to_dict(node.left),
        "right": tree_to_dict(node.right),
    }


a1 = Node("A1")
a2 = Node("A2")
h1 = Node("H1", a1, a2)
b1 = Node("B1")
h2 = Node("H2", h1, b1)


def make_data():
    rows = []
    for i in range(1000):
        yield_a = round(random.uniform(5, 55), 1)
        yield_b = round(random.uniform(5, 55), 1)
        dist = round(random.uniform(0, 1), 2)
        same_group = random.random() < 0.3
        hybrid_yield = (yield_a + yield_b) / 2 * (1 + dist * 0.3 - (0.1 if same_group else 0))
        rows.append([yield_a, yield_b, dist, int(same_group), hybrid_yield])
    return pd.DataFrame(rows, columns=["yield_a", "yield_b", "distance", "same_group", "hybrid_yield"])


def train_model():
    data = make_data()
    x = data[["yield_a", "yield_b", "distance", "same_group"]]
    y = data["hybrid_yield"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)

    test_predictions = model.predict(x_test)
    r2 = round(r2_score(y_test, test_predictions), 3)
    mae = round(mean_absolute_error(y_test, test_predictions), 2)
    print("Model trained. R2 =", r2, " average error =", mae, "q/ha")
    return model


model = train_model()


history = []


MIN_TRAINING_YIELD = 5
MAX_TRAINING_YIELD = 55


def run_model(yield_a, yield_b, distance, same_group):
    warning = None
    if not (MIN_TRAINING_YIELD <= yield_a <= MAX_TRAINING_YIELD) or \
       not (MIN_TRAINING_YIELD <= yield_b <= MAX_TRAINING_YIELD):
        warning = ("Parent yield is outside the model's training range (" +
                   str(MIN_TRAINING_YIELD) + "-" + str(MAX_TRAINING_YIELD) +
                   " q/ha). Prediction is unreliable.")

    x = pd.DataFrame([[yield_a, yield_b, distance, int(same_group)]],
                      columns=["yield_a", "yield_b", "distance", "same_group"])
    predicted_yield = round(float(model.predict(x)[0]), 2)

    mid_parent = (yield_a + yield_b) / 2
    best_parent = max(yield_a, yield_b)
    heterosis = round((predicted_yield - mid_parent) / mid_parent * 100, 1)
    vs_best = round((predicted_yield - best_parent) / best_parent * 100, 1)

    result = {
        "predicted_yield": predicted_yield,
        "heterosis_percent": heterosis,
        "vs_best_parent_percent": vs_best,
    }
    if warning:
        result["warning"] = warning
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/lines", methods=["GET"])
def get_lines():
    return jsonify(lines.data)


@app.route("/api/lines", methods=["POST"])
def add_line():
    body = request.get_json()
    name = body.get("name")
    yield_value = body.get("yield")
    if not name or yield_value is None:
        return jsonify({"error": "need name and yield"}), 400
    lines.put(name, float(yield_value))
    return jsonify(lines.data)


@app.route("/api/lines/<name>", methods=["DELETE"])
def delete_line(name):
    if name in lines.data:
        del lines.data[name]
    return jsonify(lines.data)


@app.route("/api/pedigree", methods=["GET"])
def get_pedigree():
    return jsonify(tree_to_dict(h2))


@app.route("/api/predict", methods=["POST"])
def predict_route():
    body = request.get_json()
    yield_a = float(body["yield_a"])
    yield_b = float(body["yield_b"])
    distance = float(body["distance"])
    same_group = bool(body["same_group"])

    result = run_model(yield_a, yield_b, distance, same_group)

    record = dict(result)
    record["yield_a"] = yield_a
    record["yield_b"] = yield_b
    record["distance"] = distance
    record["same_group"] = same_group
    history.append(record)

    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(history[::-1])


@app.route("/api/best", methods=["GET"])
def best_pairs():
    names = list(lines.data.keys())
    results = []
    for a, b in itertools.combinations(names, 2):
        result = run_model(lines.get(a), lines.get(b), distance=0.6, same_group=False)
        results.append({"pair": a + " x " + b, "predicted_yield": result["predicted_yield"]})
    results.sort(key=lambda r: r["predicted_yield"], reverse=True)
    return jsonify(results[:3])


@app.route("/api/genes", methods=["GET"])
def get_genes():
    return jsonify(genes.data)


@app.route("/api/gene_cross", methods=["POST"])
def gene_cross_route():
    body = request.get_json()
    genotype_a = body["genotype_a"]
    genotype_b = body["genotype_b"]
    result = punnett_square(genotype_a, genotype_b)
    return jsonify(result)


def ask_ollama(yield_a, yield_b, distance, same_group, predicted_yield, heterosis, vs_best):
    import ollama
    prompt = (
        "You are a plant breeding expert. A trained model predicted this hybrid seed result.\n"
        "Parent A yield: " + str(yield_a) + "\n"
        "Parent B yield: " + str(yield_b) + "\n"
        "Genetic distance: " + str(distance) + "\n"
        "Same heterotic group: " + str(same_group) + "\n"
        "Predicted hybrid yield: " + str(predicted_yield) + "\n"
        "Heterosis vs mid-parent: " + str(heterosis) + "%\n"
        "Vs best parent: " + str(vs_best) + "%\n"
        "In 3 short sentences, say if this looks like a good cross and why."
    )
    response = ollama.chat(model="qwen3:8b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


@app.route("/api/explain", methods=["POST"])
def explain_route():
    body = request.get_json()
    try:
        review = ask_ollama(
            body["yield_a"], body["yield_b"], body["distance"], body["same_group"],
            body["predicted_yield"], body["heterosis_percent"], body["vs_best_parent_percent"],
        )
        return jsonify({"review": review})
    except Exception as e:
        return jsonify({"error": "Could not reach Ollama. Is it running? (ollama serve) " + str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)