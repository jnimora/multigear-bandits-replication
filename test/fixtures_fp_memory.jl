# Synthetic allocation fixtures, intentionally NOT scientific benchmark data.
# n is a power of two; all transition probabilities are exactly dyadic.
function fp_memory_model(n::Integer,A::Integer)
    n>=4 && ispow2(n) || throw(ArgumentError("n must be a power of two >=4."))
    A in 1:3 || throw(ArgumentError("Allocation fixture supports A=1,2,3."))
    P=fill(1.0/n,n,n,A+1); h=zeros(n,A+1); c=zeros(n,A+1)
    for a in 0:A, i in 1:n
        delta=a/(16.0*n)
        P[i,i,a+1]+=delta
        P[i,mod1(i+1,n),a+1]-=delta
        h[i,a+1]=-16.0*i*a+a*(a+1)/2
        c[i,a+1]=a
    end
    return FiniteMultiGearModel(P,h,c)
end

function fp_memory_workspace(n,A,criterion,kind)
    model=fp_memory_model(n,A)
    kw=criterion==:discounted ? (;criterion=criterion,beta=0.75) : (;criterion=criterion)
    w=initialize_reduced_fp(model;kw...)
    # Remove original state 1 to make m<capacity and scramble physical labels.
    for _ in 1:A
        report=pivot_reduced_fp!(w,1)
        report.accepted || error("Allocation fixture's preparatory pivot failed.")
    end
    selected=n÷2
    if kind==:final
        for _ in 2:A
            report=pivot_reduced_fp!(w,selected)
            report.accepted || error("Allocation fixture's gear preparation failed.")
        end
    end
    return w,selected
end
