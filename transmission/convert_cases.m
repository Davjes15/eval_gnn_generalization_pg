% =============================================================================
% convert_cases.m  --  Step 1 of the Layer 2 pipeline (one-time grid conversion)
% =============================================================================
%
% PURPOSE
%   Convert PowerGraph-Node's raw transmission grid definitions (MATLAB/MATPOWER
%   `System.m` files) into portable MATLAB `.mat` files that pandapower can read
%   downstream via `pandapower.converter.from_mpc`.
%
% WHY THIS STEP EXISTS (design decisions D4 + D5, see
%   ../../docs/PowerGraph_to_ENGAGE_design_decisions.md)
%   - D4: we use PowerGraph's OWN `System.m` cases (not pandapower's built-in
%     IEEE cases), because PowerGraph may have tweaked the standard cases and we
%     want the four grids to be byte-identical to the ones PowerGraph trained on.
%     That makes any cross-grid difference attributable to topology, not to a
%     mismatched case file.
%   - D5: each `System.m` is a MATLAB *function* that simply returns the `mpc`
%     struct (`function mpc = System`). Running it in GNU Octave (free, no MATLAB
%     or Octave license needed) and saving the returned struct is trivial and
%     fast. We do this ONCE and commit the resulting `.mat` files, so every
%     downstream user needs only Python + pandapower -- no Octave/MATLAB at all.
%
% HOW IT CONNECTS TO THE REST OF THE PIPELINE
%   convert_cases.m  ->  transmission/cases/<CODE>.mat
%                    ->  transmission_grids.load_case()   (Step 2)
%                    ->  transmission_graph_gen.py         (Step 3)
%
% HOW TO RUN
%   From the repository root:
%       octave --no-gui --eval "cd transmission; convert_cases"
%   or from inside the transmission/ directory:
%       octave --no-gui convert_cases.m
%   Requires: GNU Octave (tested with 6.4.0).  Install locally with e.g.
%       Ubuntu/Debian:  sudo apt-get install -y octave
%       macOS (brew):   brew install octave
%
% REFERENCES
%   MATPOWER case format:  https://matpower.org/docs/ref/matpower5.0/caseformat.html
%   PowerGraph-Node:       https://github.com/PowerGraph-Datasets/PowerGraph-Node
% =============================================================================

% Path to PowerGraph-Node's power-system case folder. Override by exporting
% POWERGRAPH_NODE_DIR before launching Octave, otherwise the default below
% (relative to this script) is used.
src_root = getenv('POWERGRAPH_NODE_DIR');
if isempty(src_root)
    % Default: sibling checkout of PowerGraph-Node next to this repo.
    src_root = fullfile('..', '..', 'PowerGraph-Node-main', '13_Power_system');
end

% Output directory for the committed .mat cases (created if missing).
out_dir = fullfile('cases');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% The four transmission grids studied (see design doc, "Grids").
cases = {'IEEE24', 'IEEE39', 'IEEE118', 'UK'};

for i = 1:numel(cases)
    code    = cases{i};
    case_dir = fullfile(src_root, code);

    if ~exist(fullfile(case_dir, 'System.m'), 'file')
        error('convert_cases:missingSource', ...
              'System.m not found for %s at %s', code, case_dir);
    end

    % Put the case folder on the path so the bare `System` function resolves,
    % then remove it afterwards so the next grid's System.m is picked up.
    addpath(case_dir);
    mpc = System();              %#ok<NASGU>  the struct we want to persist
    rmpath(case_dir);

    % Save in MATLAB v7 format (readable by scipy.io.loadmat / pandapower).
    out_file = fullfile(out_dir, [code '.mat']);
    save('-v7', out_file, 'mpc');

    % Small sanity print so the conversion is auditable from the log.
    printf('Converted %-8s -> %s  (baseMVA=%g, buses=%d, branches=%d, gens=%d)\n', ...
           code, out_file, mpc.baseMVA, rows(mpc.bus), rows(mpc.branch), rows(mpc.gen));

    clear mpc;
end

printf('\nDone. Converted %d grids into %s/\n', numel(cases), out_dir);
