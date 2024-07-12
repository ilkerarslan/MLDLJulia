using HTTP, CSV, DataFrames

filelink = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data";

data = HTTP.get(filelink)
df = CSV.read(data.body, DataFrame, header=false)

using MLJ

using Nova.PreProcessing: LabelEncoder
X = df[:, 3:end] |> Matrix
y = df[:, 2]
le = LabelEncoder()
le(y)
y = le(y, :transform)
le.classes
le(["M", "B"], :transform)

using Nova.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, 
                            test_size=0.20,
                            stratify=y,
                            random_state=1)

using Nova.PreProcessing: StandardScaler
using Nova.Decomposition: PCA 
using Nova.LinearModel: LogisticRegression

sc = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression()

Xtrn |> sc |> pca |> X -> lr(X, ytrn)
ŷtst = Xtst |> sc |> pca |> lr

using Nova.Metrics: accuracy_score
accuracy_score(ytst, ŷtst)