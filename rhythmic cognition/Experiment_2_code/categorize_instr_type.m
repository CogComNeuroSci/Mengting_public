%% categorize the instruction type
function category = categorize_instr_type(instr_type)
if ismember(instr_type, [0, 1, 2, 3])
    category = 'same_signs';
elseif ismember(instr_type, [4, 5, 6, 7])
    category = 'same_side_diff_signs';
elseif instr_type >= 8 && instr_type <= 15
    category = 'different_sides';
else
    category = 'unknown';
end
end