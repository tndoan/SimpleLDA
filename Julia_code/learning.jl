module learning
import readfile
    function initialize(K, corpus, alpha)
        D = length(corpus)
        phi = ones(D, N, K) * 1/ K
        temp = copy(alpha)
        gamma = repmat(temp', D, 1)
        for i=1:D
            for j=1:K
                gamma[i, j] = gamma[i, j] + corpus[i][1] / K
            end
        end
        beta = zeros(N, K)
       return (phi, gamma, beta)
    end

    function doingEM(K, alpha, vocFile="../data/vocab.txt", dataFile="../data/ap.dat")
        # K is number of topic
        corpus = readfile.readFile(dataFile)
        D = length(corpus) # number of doc in dataset
        vocab = readfile.readVoc(vocFile)
        N = length(vocab) # number of vocabulary in dataset
        (phi, gamma, beta) = initialize(K, corpus, alpha)

        # E step
        for d = 1:D
            convergent = false
            while !convergent
                old_gamma = gamma[d, :]
                old_phi = phi[d, :, :]
                N_d = corpus[d][1]
                w = corpus[d][2]
                for n = 1:N_d
                    for i = 1:K
                        phi[d, n, i] = exp(digamma(gamma[i])) * beta[i, w[n][1]]
                    end
                    # normalize phi_n to have 1
                    s = sum(phi[d, n, :])
                    for i = 1:K
                        phi[d, n, i] = phi[d, n, i] / s
                    end
                    temp_1 = copy(phi[d, :, :])
                    temp_2 = squeeze(temp_1, 1)
                    gamma[d, :] = alpha + sum(temp_1, 1)
                end
                # check convergence of phi and gamma
                if sum(gamma - old_gamma) < threshold & sum(phi[d, :, :] - old_phi) < threshold
                    convergent = true
                end
            end
        end
        
        # M step
        #for d=1:D
        #    for i=1:K
        #        for j=1:N
        #            beta[i, j] = phi[d, 

    end
end
