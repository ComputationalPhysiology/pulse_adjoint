


geometry = Geometry(mesh=mesh, markers=markers,
                    subdomains=subdomains,
                    microstructure=microstructure)
# geometry = Geometry.from_file(geoname)


material = HolzapfelOgden(active_model=active_model,
                          parameters=material_parameters,
                          activation=activation,
                          compressible_model=compressible_model)

mech_problem = MechanicsProblem(geometry=geometry,
                                material=material, bcs=bcs)

clinical_data = ClinicalData(volume, strain, pressure)
# clinical_data = ClinicalData.from_file(dataname)

assimilator = Assimilator(mech_problem, clinical_data)

# Here we need to set the controls
# One suggestion is to make a new parameter class
# and let each possible control paramerer be an instance
# of this class. 

assimilator.assimilate()
