for %%i in (0.2 0.4 0.6 0.8) do (
	for %%j in (0.2 0.4 0.6 0.8) do (
		python four_models.py %%i %%j mnist
	)
)