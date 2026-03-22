export save_state, load_state

"""
    save_state(filename, statename, state)

save the state to disk in a hdf5 file,
several states with different names can be saved in the same file
"""
function save_state(filename::String, statename::String, state::State)
    h5open(filename, "cw") do f
        if haskey(f, statename)
            delete_object(f, statename)
        end
        g = create_group(f, statename)
        attrs(g)["type"] = state isa State{Pure} ? "Pure" : "Mixed"
        write(g, "state", state.state)
    end
    return nothing
end

"""
    load_state(filename, statename)

load a state previously saved by `save_sate`
"""
function load_state(filename::String, statename::String)
    error("load_state is not implemented")
end
