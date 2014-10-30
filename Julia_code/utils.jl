function readFile(filename="../data/ap.dat")
    # read the data file
    # return list of document whose format is 1st value is the number of words and following array of pair (word id, its frequency)
    result = (Int64, [])[]
    open(filename, "r") do f
        for line in eachline(f)
            comp = split(line, " ")
            numWords = int(comp[1])
            bagOfWords = comp[2:size(comp)[1]]
            wordInfo = []
            for word in bagOfWords
                temp = split(word, ":")
                if length(temp) != 2
                    continue
                end
                # temp[1] is the word id
                # temp[2] is the number of times that this word appears in document
                push!(wordInfo, (int(temp[1]), int(temp[2])))
            end
            tuple = (numWords, wordInfo)
            push!(result, tuple)
        end
    end
    return result
end

function readVoc(filename="../data/vocab.txt")
    # read the list of vocabulary
    result = ASCIIString[]
    open(filename, "r") do f
        for line in eachline(f)
            vocab = split(line, "\n")[1]
            push!(result, vocab)
        end
    end
    return result
end

function readMatrix(filename="../data/ap.dat", voc)
    # read data file
    # filename name of file contain LDA-C format
    # voc list of vocabulary
    
    corpus = readFile(filename)
    return convertToMatrix(corpus, voc)

end

function convertToMatrix(corpus, voc)
    # voc vocabulary list
    # convert corpus which is returned by readFile to matrix D x N
    # D is number of document
    # N is number of words 

    N = length(voc) # number of words
    D = length(corpus) # number of documents
    result = zeros(D, N)

    for d=1:D
        doc = corpus[d]
        numWords = doc[0]
        wordInfo = doc[1]
        for word in wordInfo
            wordId = word[0]
            occurrence = word[1]
            result[d, wordId] = occurrence
        end
    end

    return result
end
