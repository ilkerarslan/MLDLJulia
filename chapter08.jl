using ProgressMeter
using DataFrames
using CSV
using Random

#=
# Combine all files in a single dataframe
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

df = CSV.read("data/movie_data.csv", DataFrame)

using NovaML.FeatureExtraction
count = CountVectorizer()
docs = ["The sun is shining", "The weather is sweet", "The sun is shining, the weather is sweet, and one and one is two"]
bag = count(docs)

count.vocabulary