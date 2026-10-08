include("frozen_common.jl")
function worker(jobfile,out)
    VERSION==v"1.13.0" || error("Frozen references require Julia 1.13.0")
    Threads.nthreads()==1 || error("Use one compute thread")
    job=TOML.parsefile(jobfile);mkpath(out);progress=joinpath(out,"progress.toml")
    old=job["primitives"]
    job["eligible_archive"] || error("Historically unresolved case: no published timing")
    initial=reconstruct(old,job["kind"])
    input_identity(old,job["kind"],initial.input)["passed"] || error("Primitive hash mismatch")
    rr=prepare_run(initial.input,initial.family,job,"FP")
    K=length(rr.log.state);targets=unique([0,(K-1)÷2,K-1])
    while rr.status in (:ready,:running)
        if rr.log.k in targets
            audit_performance_state(rr).passed || error("Fresh reference audit failed")
        end
        step_performance!(rr)
    end
    reference=snapshot(rr);reference.complete || error("Fresh reference incomplete")
    initial=nothing;rr=nothing;GC.gc()
    variants=String.(job["variants"]);clock=TimingClocks()
    small=reconstruct(old,job["kind"];warmup=true)
    for v in variants,_ in 1:2
        timing_instrument(clock,total_execution,small.input,small.family,job,v)
        timing_instrument(clock,phase_execution,small.input,small.family,job,v)
    end
    small=nothing;GC.gc()
    generation=@timed reconstruct(old,job["kind"]);x=generation.value
    identity=input_identity(old,job["kind"],x.input)
    identity["passed"] || error("Reconstructed input differs from frozen primitive bytes")
    findall(x.input.model.controllable)==reference.states || error("Frozen controllable-state labels differ")
    record=Dict{String,Any}("job"=>job,"model"=>old["model"],"status"=>"validating",
        "input_identity"=>identity,"generation_and_admission_seconds"=>generation.time,
        "generation_allocated_bytes"=>generation.bytes,"clock"=>timing_clock_metadata(clock),
        "environment"=>Dict("julia"=>string(VERSION),"blas"=>string(BLAS.get_config()),
            "blas_threads"=>BLAS.get_num_threads(),"compute_threads"=>Threads.nthreads(),"cpu"=>Sys.CPU_NAME),
        "validation"=>Any[],"measurements"=>Any[],"archived_validation_reused"=>false,
        "validation_reuse_condition"=>"Primitive hashes checked; FP reference regenerated with three fresh audits; each blocked variant also receives three fresh audits",
        "scope"=>"Separate end-to-end and phase executions; input reconstruction/admission and audits outside timings; all selection/fallback work inside timings")
    save_record(joinpath(out,"result.toml"),record)
    allowed=Dict(v=>true for v in variants);expected=Dict{String,Any}()
    for v in variants
        v in ("FP","M") && continue
        GC.gc();save_record(progress,Dict("stage"=>"validation","variant"=>v))
        z=@timed checked_blocked(x.input,x.family,job,v,progress);checked=z.value;s=checked.snapshot
        ok=checked.passed && agreement(s,reference)
        push!(record["validation"],Dict("variant"=>v,"passed"=>ok,"algorithm_status"=>string(s.status),
            "audits"=>checked.audits,"agreement_errors"=>errors(s,reference),"counts"=>s.counts,
            "blocking"=>checked.blocking,"seconds"=>z.time))
        allowed[v]=ok;expected[v]=stable_counts(s)
        if !ok;open(io->serialize(io,s),joinpath(out,v*"_rejected_snapshot.bin"),"w");end
        save_record(joinpath(out,"result.toml"),record)
        println(job["id"]," ",v," validation ",ok);flush(stdout)
    end
    function measure(v,rep)
        allowed[v] || return
        for phasekind in ("end_to_end","phases")
            save_record(progress,Dict("stage"=>"timing","variant"=>v,"repetition"=>rep,"execution"=>phasekind))
            GC.gc()
            f=phasekind=="end_to_end" ? total_execution : phase_execution
            t=timing_instrument(clock,f,x.input,x.family,job,v)
            s=phasekind=="end_to_end" ? t.measured.value : t.measured.value.snapshot
            matched=agreement(s,reference)
            counts=stable_counts(s)
            haskey(expected,v) ? (matched &= counts==expected[v]) : (expected[v]=counts)
            row=clocks_record(t)
            row["variant"]=v;row["repetition"]=rep;row["execution"]=phasekind
            row["matched_reference"]=matched;row["counts"]=s.counts;row["agreement_errors"]=errors(s,reference)
            row["algorithm_status"]=string(s.status)
            row["eligible"]=matched && row["compiled_seconds"]==0 && !row["diagnostic_flag"]
            if phasekind=="phases"
                q=t.measured.value
                row["initialization_seconds"]=q.initialization;row["index_loop_seconds"]=q.index_loop
                row["output_seconds"]=q.output;row["blocking"]=q.blocking
            end
            push!(record["measurements"],row);record["peak_rss_bytes"]=Int(Sys.maxrss())
            record["status"]="timing";save_record(joinpath(out,"result.toml"),record)
            println(job["id"]," ",v," rep",rep," ",phasekind," ",round(row["seconds"];digits=3),"s eligible ",row["eligible"]);flush(stdout)
            if !matched
                allowed[v]=false
                open(io->serialize(io,s),joinpath(out,v*"_rejected_snapshot.bin"),"w")
                return
            end
        end
    end
    order=circshift(variants,mod(job["sequence"]-1,length(variants)))
    for rep in 1:2,v in (rep==1 ? order : reverse(order));measure(v,rep);end
    # A single prospective replacement attempt is allowed only for timing
    # contamination/compilation, never for numerical rejection. Retain all rows.
    for v in order
        allowed[v] || continue
        any(kind->count(r->r["variant"]==v && r["execution"]==kind && r["eligible"],record["measurements"])<2,
            ("end_to_end","phases")) && measure(v,3)
    end
    coverage=Dict{String,Any}()
    for v in variants
        d=Dict{String,Any}(kind=>count(r->r["variant"]==v && r["execution"]==kind && r["eligible"],record["measurements"])
               for kind in ("end_to_end","phases"))
        d["valid"]=allowed[v] && all(n->n>=2,values(d));coverage[v]=d
    end
    record["coverage"]=coverage
    record["status"]=all(v->v["valid"],values(coverage)) ? "accepted" : "incomplete_comparison"
    record["peak_rss_bytes"]=Int(Sys.maxrss());save_record(joinpath(out,"result.toml"),record)
end
if abspath(PROGRAM_FILE)==@__FILE__
    try
        worker(ARGS[1],abspath(ARGS[2]))
    catch e
        save_record(joinpath(ARGS[2],"error.toml"),Dict("error"=>sprint(showerror,e,catch_backtrace())))
        rethrow()
    end
end
