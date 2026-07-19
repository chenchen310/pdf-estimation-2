"""GDS -> BBP image prediction for die-to-database reference generation.

Stage-0 model: the frame is a linear combination of band-limited region
densities (Boolean layer combos x local pattern orientation), because every
drawn feature is far below the optical resolution limit -- the image only
sees locally averaged effective reflectance.
"""
