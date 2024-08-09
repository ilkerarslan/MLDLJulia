using CSV, DataFrames
using Random

#=
# Combine all files in a single dataframe
using ProgressMeter
basepath = "aclImdb"
labels = Dict("pos" => 1, "neg" => 0)
p = Progress(50000, desc="Processing files: ", barglyphs=BarGlyphs("[=> ]"))
df = DataFrame(review = String[], sentiment = Int[])
for s in ["test", "train"]
    for l in ["pos", "neg"]
        path = joinpath(basepath, s, l)
        for file in sort(readdir(path))
            txt = open(joinpath(path, file), "r") do infile
                read(infile, String)
            end
            push!(df, (txt, labels[l]))
            next!(p)
        end
    end
end
df
CSV.write("data/movie_reviews.csv", df)
Random.seed!(0)
idx = shuffle(1:nrow(df));
df = df[idx, :]
CSV.write("data/movie_data.csv", df)
=#

using NovaML.FeatureExtraction
docs = ["The sun is shining", 
        "The weather is sweet", 
        "The sun is shining, the weather is sweet, and one and one is two"];

countvec = CountVectorizer()
bag = countvec(docs)
countvec.vocabulary
countvec(bag, type=:inverse_transform)

tf = bag
df = [1, 3, 1, 2, 2, 2, 3, 1, 2]
idf = log.(4 ./ (1 .+ df))
idf = idf .+ 1
tfidf = tf .* idf'
row_norms = sqrt.(sum(tfidf.^2, dims=2))
tfidf = tfidf ./ max.(row_norms, eps())

df = CSV.read("data/movie_data.csv", DataFrame)
text = df[2, :review]

using Unicode

function preprocessor(text::String)
    # Remove HTML tags
    text = replace(text, r"<[^>]*>" => "")
    
    # Find emoticons
    emoticons = [m.match for m in eachmatch(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)]
    
    # Convert to lowercase, remove non-word characters, and add emoticons back
    text = lowercase(text)
    text = replace(text, r"[\W]+" => " ")
    text *= " " * join(replace.(emoticons, "-" => ""), " ")
    
    # Normalize unicode characters and strip leading/trailing whitespace
    return strip(Unicode.normalize(text, stripmark=true))
end

# Test the function
processed_text = preprocessor(text)
println("Original text: ", text)
println("Processed text: ", processed_text)
preprocessor("</a>This :) is :( a test :-)!")

df.review = preprocessor.(df.review);
df

function tokenizer(text)
    return split(text)
end
tokenizer("runners like running and thus they run")

using Snowball
stemmer = Stemmer("english")
stem(stemmer, "running")

using Languages, WordTokenizers
stop_words = stopwords(Languages.English())
text = "runners like running and thus they run"
tokens = tokenize(text)
[w for w in tokenize(text) if w ∉ stop_words]

#Training a logistic regression model for document classification
Xtrn, Xtst = df[1:25000, :review], df[25001:end, :review];
ytrn, ytst = df[1:25000, :sentiment], df[25001:end, :sentiment]

using NovaML.ModelSelection: GridSearchCV
using NovaML.Pipelines: pipe
using NovaML.LinearModel: LogisticRegression

using NovaML.FeatureExtraction
tfidf = TfidfVectorizer()
docs = ["The sun is shining", 
        "The weather is sweet", 
        "The sun is shining, the weather is sweet, and one and one is two"]

result = tfidf(docs)
tfidf.vocabulary
