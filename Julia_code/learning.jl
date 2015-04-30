require("utils")

function calculateLikelihood(matrix, beta, gammas)
    llh = 0
    D, V = size(matrix) # number of documents

    for i = 1:D
        g = gammas[:, d]  / sum(gammas[:, d]) # normalize
        for j = 1:V
            if matrix[i, j] != 0
                llh += matrix[i, j] * dot(g, beta[:, j])
            end
        end
    end

    return llh
end

function EStep(K, alpha, beta, docVector)
    # doing E-step
    # K is number of topic
    # alpha, beta are parameters
    # docVector: term vector of doc 1 x N

    # init phi and gamma
    wIds = find(docVector) # word id of term in document
    N = length(wIds)
    phi = ones(N, K) * 1 / K
    gamma = copy(alpha) + N / K # size of gamma K x 1

    # loop until convergence
    convergent = false
    threshold = 0.01 # parameter for checking convergence
    counter = 1
    println("Estep")
    while ~convergent
        println(counter)
        oldPhi = copy(phi)
        oldGamma = copy(gamma)

        # update phi
        e_dig = exp(digamma(gamma))
        phi = beta[:, wIds] .* repmat(e_dig, 1, N)
        phi = phi ./ repmat(sum(phi, 2), 1, N)

        # update gamma
        println(alpha)
        println(sum(phi, 1))
        gamma = alpha + sum(phi, 2)'

        # checking convergence
        m = max(abs(phi - oldPhi), abs(gamma - oldGamma))
        if m < threshold
            convergent = true
        end
        counter += 1
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


function doingEM(K, vocFile="../data/vocab.txt", dataFile="../data/ap.dat")
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
    # init alpha and beta
    alpha = rand(K, 1)
    b = rand(K, D)
    beta = broadcast(*, b, 1 ./ sum(b, 1)) # normalize

    counter = 1

    pllh = 0 # previous likelihood
    threshold = 0.001

    while ~convergent
        # loop until convergence
        println(counter)

        # E-step
        for d=1:D
            # doing for each document in corpus
            (subPhi, subGamma) = EStep(K, alpha, beta, matrix[d, :])
            phi[d, :, :] = subPhi
            gamma[:, d] = subGamma
        end

        # M-step
        (alpha, beta) = MStep(K, N, phi, gamma, matrix)

        # checking convergence
        llh = calculateLikelihood(matrix, beta, gamma)
        if abs(pllh - llh) < threshold
            convergent = true
        else
            pllh = llh
            println(llh)
        end
        counter += 1
    end
end
