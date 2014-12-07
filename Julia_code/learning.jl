require("utils")

# TODO: function to calculate likelihood
function calculateLikelihood(alpha, beta)

    return 0
end

function EStep(K, N, d, alpha, beta, docVector)
    # doing E-step
    # K is number of topic
    # N is total number of words
    # d is the index of document
    # alpha, beta are parameters
    # docVector: term vector of doc 1 x N

    # init phi and gamma
    phi = ones(N, K) * 1 / K
    gamma = copy(alpha) + N / K # size of gamma K x 1
    wIds = find(docVector) # word id of term in document d

    # loop until convergence
    convergent = false
    threshold = 0.01 # parameter for checking convergence
    while ~convergent
        oldPhi = copy(phi)
        oldGamma = copy(gamma)

        # update phi
        e_dig = exp(digamma(gamma))
        phi = beta[:, d] .* repmat(e_dig, 1, N)
        phi = phi ./ repmat(sum(phi, 2), 1, N)

        # update gamma
        gamma = alpha + sum(phi, 1)

        # checking convergence
        m = max(abs(phi - oldPhi), abs(gamma - oldGamma))
        if m < threshold
            convergent = true
        end
    end

    # return result
    return (phi, gamma)
end

function MStep(K, D, N, phi, gamma, matrix)
    # K is number of topic
    # D is total number of documents
    # N is total number of words
    beta = zeros(K, N)
    alpha = zeros(K, 1)
    
    # update beta
    for d=1:D
        for n=1:N
            if matrix[d, n] > 0 # this word belongs to this doc
                # squeeze to convert to vector
                beta[:, n] += squeeze(phi[d, n, :], 1)'
            end
        end
    end
    #normalize beta
    beta /= repmat(sum(beta, 2), 1, N)
    
    # update alpha
    convergent = false
    threshold = 0.001
    while ~convergent
        oldAlpha = alpha

        # gradient of alpha
        g = D * (digamma(sum(oldAlpha)) - digamma(oldAlpha))
        g += sum(digamma(gamma), 2) - sum(digamma(sum(gamma, 1)))

        # Hessian matrix of alpha
        h = D * (trigamma(sum(oldAlpha)) - eye(K) * trigamma(oldAlpha))

        # calculate alpha
        # TODO could be better if we use optimization from paper
        alpha = oldAlpha - inv(h) * g

        # check convergence
        if sum(abs(alpha - oldAlpha)) < threshold
            convergent = true
        end
    end

    # return alpha, beta
    return (alpha, beta)
end


function doingEM(K, alpha, beta, vocFile="../data/vocab.txt", dataFile="../data/ap.dat")
    # K is number of topic
    # D number of documents
    # N total number of words
    # alpha is parameter of Dirichlet distribution (size K x 1)
    # beta is matrix K x D
    corpus = readFile(dataFile)
    D = length(corpus) # number of doc in dataset
    vocab = readVoc(vocFile)
    N = length(vocab) # number of vocabulary in dataset
    matrix = convertToMatrix(corpus, vocab) # size D x N

    convergent = false
    # create phi and gamma to allocate memory only
    phi = zeros(D, N, K)
    gamma = zeros(K, D)

    # old alpha, beta
    alpha0 = alpha
    beta0 = beta

    threshold = 0.001

    while ~convergent
        # loop until convergence

        # E-step
        for d=1:D
            # doing for each document in corpus
            (subPhi, subGamma) = EStep(K, N, d, alpha, beta, matrix[d, :])
            phi[d, :, :] = subPhi
            gamma[:, d] = subGamma
        end

        # M-step
        (alpha, beta) = MStep(K, N, phi, gamma, matrix)

        # checking convergence
        errRate = sum(abs(alpha - alpha0)) + sum(abs(beta - beta0))
        if errRate < threshold
            convergent = true
        else
            alpha0 = alpha
            beta0 = beta
        end
    end
end
