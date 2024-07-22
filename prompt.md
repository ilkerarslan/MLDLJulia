Can you implement the sklearn.model_selection.GridSearchCV for NovaML. Implementation of GridSearchCV in sklearn is attached.
And also consider the following use case in Python.

```Python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svcC': param_range,
               'svckernel': ['linear']},
              {'svcC': param_range,
               'svcgamma': param_range,
               'svckernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  refit=True,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.bestscore)
# 0.9846153846153847
print(gs.bestparams)
# {'svcC': 100.0, 'svcgamma': 0.001, 'svckernel': 'rbf'}
```

I want to implement this as below in NovaML:

```Julia
using NovaML.ModelSelection: GridSearchCV
using NovaML.SVM: SVC
using NovaML.PreProcessing: StandardScaler
using NovaML.Pipelines: Pipe
using NovaML.Metrics: accuracy_score

scaler = StandardScaler()
svc = SVC()
pipe_svc = Pipe(scaler, svc)
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    Dict(
        (svc, :C) => param_range,
        (svc, :kernel) => [:linear]
    ),
    Dict(
        (svc, :C) => param_range,
        (svc, :gamma) => param_range,
        (svc, :kernel) => [:rbf]
    )
]

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring=accuracy_score,
    cv=10,
    refit=true,
    n_jobs=-1
)

gs(X_train, y_train)
println(gs.bestscore)
println(gs.bestparams)
```

Some notes:
* Use parallel threading via n_jobs. Make use of Julia's builtin threading capabilities instead of an external package for this.
* I am also open to your suggestions that may be useful.
* Keep NovaML implementation principles in mind.