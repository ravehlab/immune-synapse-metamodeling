function ptcr_array = tcrPhosphorylation(...
    tcr_array,alck_array,phos_threshold)

% multiply alck distribution by rand array
random_alck_array = alck_array.*abs(randn(size(alck_array)));
normalyzed_random_alck_array = ...
    random_alck_array/max(random_alck_array(:));

phos_ability_array = zeros(size(alck_array));
phos_ability_array(normalyzed_random_alck_array > ...
    phos_threshold) = 1;

ptcr_array = tcr_array.*phos_ability_array;

end


