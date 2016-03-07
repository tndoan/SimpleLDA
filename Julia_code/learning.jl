function calculateLikelihood(matrix, beta, gammas)
    llh = 0
    D, V = size(matrix) # number of documents

    for d = 1:D
        g = gammas[:, d]  / sum(gammas[:, d]) # normalize
        for j = 1:V
            if matrix[d, j] != 0
                llh += matrix[d, j] * log(dot(g, beta[:, j]))
            end
        end
    end

    return llh
end

function EStep(K, alpha, beta, docVector, maxIter=10)
    # doing E-step
    # K is number of topic
    # alpha, beta are parameters
    # docVector: term vector of doc 1 x N
    # docVector[i] = j means i-th word has j occurrence in corpus

    # init phi and gamma
    wIds = find(docVector) # word id of term in document
    N = length(wIds)
    phi = (ones(N, K) * 1 / K)'
    gammas = copy(alpha) + N / K # size of gamma K x 1

    # loop until convergence
    convergent = false
    threshold = 0.01 # parameter for checking convergence
    counter = 1
    
    while ~convergent
        oldPhi = copy(phi)
        oldGamma = copy(gammas)

        # update phi
        e_dig = exp(digamma(gammas))
        phi = beta[:, wIds] .* e_dig
        phi = phi ./ sum(phi, 1)

        # update gamma
        gammas = alpha + sum(phi, 2)

        # checking convergence
        m = max(sum(abs(phi - oldPhi)), sum(abs(gammas - oldGamma)))
        if m < threshold || counter == 10
            convergent = true
        end
        counter += 1
    end

    # return result
    return (phi, gammas)
end

function updateBeta(K, phi, matrix)
    D, N = size(matrix)
    beta = zeros(K, N)
    
    # update beta
    for d=1:D
        wIds = find(matrix[d, :])
        beta[:, wIds] += (diagm(vec(matrix[d, wIds])) * phi[d]')'
    end
    #normalize beta
    beta = broadcast( * , beta, 1./sum(beta, 1))
    return beta
end

function updateAlpha(K, gammas, maxIter=10)
    # K is number of topic
    # D is total number of documents
    # N is total number of words
    # phi is 
    # gammas is K x D matrix
    # matrix is (
    
    #alpha = zeros(K, 1) + 1
    (K, D) = size(gammas)
    alpha = mean(gammas, 2) ./ K
    # update alpha
    convergent = false
    threshold = 0.01
    counter = 1
    pg = sum(digamma(gammas),2) - sum(digamma(sum(gammas,1)));

    while ~convergent
        #L = L_α(alpha, gammas)
        #println("MStep: ", counter, ":L:", L, ":alpha:", alpha)
        oldAlpha = copy(alpha)
        
        alpha0 = sum(alpha)
        g = D * (digamma(alpha0) - digamma(alpha)) + pg
        h = -1 ./ trigamma(alpha)
        hgz = h' * g / (1 / trigamma(alpha0) + sum(h))

        for i=1:K
            alpha[i] = alpha[i] - h[i] * ( g[i] - hgz[1, 1]) / D
        end

        # check convergence
        if sum(abs(alpha - oldAlpha)) < threshold || counter == maxIter
            convergent = true
        end
        counter +=1 
    end

    return alpha
end

function L_α(α, γs)
    (K, D) = size(γs)
    L = D * (  log(gamma(sum(α))) - sum(log(gamma(α))))
    for d=1:D
        for i=1:K
            L += ( α[i] - 1 ) * ( digamma(γs[i, d]) - digamma(sum(γs[:, d])))
        end
    end
    return L
end


function doingEM(K, vocFile="../data/testvocab.txt", dataFile="../data/test.dat")
    # K is number of topic
    # D number of documents
    # N total number of words
    # alpha is parameter of Dirichlet distribution (size K x 1)
    # beta is matrix K x D
    corpus = readFile(dataFile)
    D = length(corpus) # number of doc in dataset
    println(D)
    vocab = readVoc(vocFile)
    N = length(vocab) # number of vocabulary in dataset
    println(N)
    matrix = convertToMatrix(corpus, vocab) # size D x N

    convergent = false
    # create phi and gamma to allocate memory only
    phi = Dict()
    gammas = zeros(K, D)
    # init alpha and beta
    alpha = rand(K, 1)
    alpha = alpha ./ sum(alpha) # normalize alpha
    b = rand(K, N)
    beta = broadcast(*, b, 1 ./ sum(b, 1)) # normalize

    counter = 1

    pllh = 0 # previous likelihood
    threshold = 0.01

    while ~convergent
        # loop until convergence

        # E-step
        #println("E step")
        for d = 1:D
            # doing for each document in corpus
            (subPhi, subGamma) = EStep(K, alpha, beta, matrix[d, :])
            phi[d] = subPhi
            gammas[:, d] = subGamma
        end

        # M-step
        #println("M Step")
        beta = updateBeta(K, phi, matrix)
        alpha = updateAlpha(K, gammas)

        # checking convergence
        llh = calculateLikelihood(matrix, beta, gammas)
        println("iter:", counter,"\tLLH:", llh)
        #if abs(pllh - llh) < threshold
        if counter == 10
            convergent = true
        #else
        #    pllh = llh
        #    println(llh)
        end
        counter += 1
    end
end
