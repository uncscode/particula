"""
Run all the machine learning trainings
"""

from particula.data.process.ml_analysis import generate_and_train_2mode_sizer


# run the 2 mode sizer training
print("Running 2 mode sizer training")
generate_and_train_2mode_sizer.train_network_and_save()
