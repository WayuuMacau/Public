import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, SGDRegressor, QuantileRegressor, TweedieRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, ConstantKernel
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestCentroid

# For MCMC Neural Network (might need installation)
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive
    import jax.numpy as jnp
    import jax.random as random
    NUMPYRO_AVAILABLE = True
except ImportError:
    print("Warning: numpyro not installed. MCMC Neural Network will be skipped.")
    print("Install with: pip install numpyro jax")
    NUMPYRO_AVAILABLE = False

# For ARIMA & GARCH (might need installation)
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    print("Warning: arch package not installed. ARIMA & GARCH will be skipped.")
    print("Install with: pip install arch")
    ARCH_AVAILABLE = False

# For Prophet (might need installation)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Warning: prophet not installed. Prophet model will be skipped.")
    print("Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# For TensorFlow Probability (might need installation)
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    print("Warning: tensorflow_probability not installed. Bayesian Structural Time Series will be skipped.")
    print("Install with: pip install tensorflow tensorflow_probability")
    TFP_AVAILABLE = False


class DirectionalAccuracyMixin:
    """Mixin class for directional accuracy evaluation"""

    def directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy of predictions"""
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        dir_true = np.sign(np.diff(np.append([y_true_np[0]], y_true_np)))
        dir_pred = np.sign(np.diff(np.append([y_true_np[0]], y_pred_np)))

        accuracy = np.mean(dir_true == dir_pred)
        return accuracy


class TimeWeightedRegressor(LassoCV):
    """Meta-regressor with time decay weights"""

    def fit(self, X, y, **kwargs):
        """Fit with time-weighted samples"""
        time_weights = np.linspace(0.5, 1.0, len(y))
        return super().fit(X, y, sample_weight=time_weights)


class FinancialStackingEnsembleRegressor(BaseEstimator, RegressorMixin, DirectionalAccuracyMixin):
    """Enhanced stacking ensemble regressor for financial predictions"""

    def __init__(self, n_splits=5, random_state=42, meta_learner='gbm', uncertainty=True,
                 feature_engineering=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.meta_learner = meta_learner
        self.uncertainty = uncertainty
        self.feature_engineering = feature_engineering

        # Initialize base models with financial-oriented algorithms
        self.base_models = []

        # 1. Gaussian Process with optimized kernels for financial data
        from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel

        kernel = DotProduct(sigma_0_bounds=(1e-10, 10.0)) + WhiteKernel(noise_level=1)

        self.base_models.append(('gpr', GaussianProcessRegressor(
            kernel=kernel,
            random_state=0,
            alpha=0.0001,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=500,
            normalize_y=True
        )))

        # 2. SGD Regressor with Huber loss for robustness to outliers
        self.base_models.append(('sgd', SGDRegressor(
            loss='huber',
            penalty='elasticnet',
            max_iter=2000,
            tol=1e-3,
            learning_rate='adaptive',
            eta0=0.001,
            random_state=random_state
        )))

        # 3. Partial Least Squares with more components for complex relationships
        self.base_models.append(('pls', PLSRegression(n_components=5)))

        # 4. Nearest Centroid with Manhattan distance
        self.base_models.append(('nc', NearestCentroid(metric='manhattan')))

        # 5. Quantile Regressor (median prediction)
        self.base_models.append(('qr', QuantileRegressor(quantile=0.5, alpha=0.1, solver='highs')))

        # 6. Tweedie Regressor for handling skewed distributions
        self.base_models.append(('tweedie', TweedieRegressor(power=1.5, link='auto', alpha=0.00001, fit_intercept=True, solver='lbfgs', max_iter=10000, tol=0.0001, warm_start=False, verbose=0)))

        # 7. Advanced GARCH model (if available)
        if ARCH_AVAILABLE:
            self.base_models.append(('garch', 'advanced_garch'))

        # 8. Prophet model for seasonality and trend (if available)
        if PROPHET_AVAILABLE:
            self.base_models.append(('prophet', 'prophet_model'))

        # 9. Bayesian Structural Time Series (if available)
        if TFP_AVAILABLE:
            self.base_models.append(('bsts', 'bayesian_sts'))

        # Choose meta-learner based on input parameter
        if meta_learner == 'time_weighted':
            self.meta_model = TimeWeightedRegressor(cv=3)
        elif meta_learner == 'gbm':
            # Define param_grid here
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'loss': ['huber', 'ls']
            }

            # Create a GridSearchCV object
            grid_search = GridSearchCV(
                GradientBoostingRegressor(random_state=self.random_state),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1  # Use all available cores
            )

            # Use a subset of data for hyperparameter tuning
            X_train_sample = np.random.rand(100, 10)  # Example placeholder
            y_train_sample = np.random.rand(100)  # Example placeholder

            # Fit GridSearchCV
            grid_search.fit(X_train_sample, y_train_sample)

            # Get the best parameters
            best_params = grid_search.best_params_
            print("Best parameters for GradientBoostingRegressor:", best_params)

            # Initialize meta-learner with the best parameters
            self.meta_model = GradientBoostingRegressor(
                n_estimators=best_params['n_estimators'],
                learning_rate=best_params['learning_rate'],
                max_depth=best_params['max_depth'],
                loss=best_params['loss'],
                random_state=self.random_state
            )
        else:
            self.meta_model = LassoCV(cv=3, random_state=random_state)

        self.scaler = StandardScaler()
        self.models_need_scaling = ['gpr', 'sgd', 'pls', 'nc', 'qr']
   
    def _add_financial_features(self, X, is_training=True):
        if not self.feature_engineering:
            return X

        X_enhanced = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0)
        features_to_concat = [X_enhanced]

        for col in X_enhanced.columns:
            # Add MACD (Moving Average Convergence Divergence)
            ema_12 = X_enhanced[col].ewm(span=12, adjust=False).mean()
            ema_26 = X_enhanced[col].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            features_to_concat.append(macd.rename(f'{col}_macd'))
            features_to_concat.append(signal.rename(f'{col}_macd_signal'))

        X_enhanced = pd.concat(features_to_concat, axis=1).copy()
        X_enhanced = X_enhanced.replace([np.inf, -np.inf], 0).fillna(0)
        if is_training:
            self.engineered_features = X_enhanced.columns.tolist()
        return X_enhanced.values
 

#2Ensemble.py
    def _fit_mcmc_neural_net(self, X, y):
        """Fit enhanced MCMC Neural Network using numpyro"""
        key = random.PRNGKey(self.random_state)
       
        # Define a Bayesian neural network with two hidden layers
        def model(X, y=None):
            n_features = X.shape[1]
            n_hidden = 10
           
            # Sample weights and biases for first layer
            w1 = numpyro.sample('w1', dist.Normal(0, 1).expand([n_features, n_hidden]))
            b1 = numpyro.sample('b1', dist.Normal(0, 1).expand([n_hidden]))
           
            # Sample weights and bias for output layer
            w2 = numpyro.sample('w2', dist.Normal(0, 1).expand([n_hidden, 1]))
            b2 = numpyro.sample('b2', dist.Normal(0, 1))
           
            # Heteroskedastic noise (different noise for different inputs)
            sigma = numpyro.sample('sigma', dist.HalfCauchy(1))
           
            # Forward pass through network
            h1 = jnp.tanh(jnp.dot(X, w1) + b1)  # Hidden layer with tanh activation
            mean = jnp.dot(h1, w2).flatten() + b2  # Output layer
           
            # Likelihood
            with numpyro.plate('data', X.shape[0]):
                numpyro.sample('obs', dist.StudentT(df=5, loc=mean, scale=sigma), obs=y)  # t-distribution for fat tails
       
        # Setup NUTS sampler and MCMC
        nuts_kernel = NUTS(model, target_accept_prob=0.9)
        mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
       
        # Run inference
        mcmc.run(key, jnp.array(X), jnp.array(y))
       
        # Store samples and model
        self.mcmc_samples = mcmc.get_samples()
        self.mcmc_model = model
       
        return self
   
    def _predict_mcmc_neural_net(self, X):
        """Predict using fitted MCMC Neural Network with uncertainty"""
        predictive = Predictive(self.mcmc_model, self.mcmc_samples)
        predictions = predictive(random.PRNGKey(0), jnp.array(X))
       
        # Get mean prediction
        mean_pred = np.array(predictions['obs'].mean(axis=0))
       
        if self.uncertainty:
            # Calculate prediction intervals
            lower = np.percentile(predictions['obs'], 5, axis=0)
            upper = np.percentile(predictions['obs'], 95, axis=0)
            self.last_uncertainty = {'lower': lower, 'upper': upper}
           
        return mean_pred
   
    def _fit_advanced_garch(self, y):
        """Fit GJR-GARCH model with skewed t-distribution"""
        y_rescaled = y * 100 if np.max(np.abs(y)) < 1 else y
        returns = pd.Series(y_rescaled).pct_change().dropna().values

        #GARCH
        p_values = range(1, 3)
        q_values = range(1, 3)

        best_aic = float('inf')
        best_order = None
        best_model = None

        for p in p_values:
            for q in q_values:
                try:
                    garch_model = arch_model(returns, vol='Garch', p=p, q=q)
                    garch_result = garch_model.fit(disp='off')
                    aic = garch_result.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, q)
                        best_model = garch_result
                except Exception as e:
                    continue

        self.garch_model = arch_model(
            returns,
            vol='GARCH',
            p=best_order[0], o=1, q=best_order[1],
            mean='AR',
            lags=2,
            dist='skewt'
        )
        self.garch_res = self.garch_model.fit(disp='off')
        return self

    def _predict_advanced_garch(self, steps=1):
        """Predict using fitted GARCH model with volatility information"""
        forecast = self.garch_res.forecast(horizon=steps)
        mean_forecast = forecast.mean.iloc[-1].values[0] / 100  # Rescale back
        return mean_forecast
   
    def _fit_prophet(self, dates, y):
        """Fit Facebook Prophet for financial time series"""
        if not isinstance(dates, pd.DatetimeIndex):
            # Create a default date range if dates aren't provided
            dates = pd.date_range(end='today', periods=len(y))
            
        df = pd.DataFrame({'ds': dates, 'y': y})
        
        # Configure Prophet for financial data
        self.prophet_model = Prophet(
            changepoint_prior_scale=0.05,  # More flexible trend changes
            seasonality_prior_scale=10.0,  # Stronger seasonality
            seasonality_mode='multiplicative',  # For financial data
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # Add higher frequency seasonality if we have enough data
        if len(y) > 30:
            self.prophet_model.add_seasonality(name='monthly', period=30, fourier_order=5)
        
        # Fit the model
        self.prophet_model.fit(df)
        return self

    def _predict_prophet(self, future_periods=1):
        """Generate Prophet forecast with uncertainty intervals"""
        future = self.prophet_model.make_future_dataframe(periods=future_periods)
        forecast = self.prophet_model.predict(future)
        
        # Get the predictions for future periods
        predictions = forecast['yhat'].iloc[-future_periods:].values
        
        if self.uncertainty:
            # Store uncertainty bounds
            self.prophet_uncertainty = {
                'lower': forecast['yhat_lower'].iloc[-future_periods:].values,
                'upper': forecast['yhat_upper'].iloc[-future_periods:].values
            }
            
        return predictions[-1] if future_periods == 1 else predictions

    def _fit_bayesian_sts(self, y):
        """Fit Bayesian structural time series model"""
        # Convert to tensor
        y_tensor = tfp.sts.regularize_series(y)
        
        # Create model components
        trend = tfp.sts.LocalLinearTrend(observed_time_series=y_tensor)
        seasonal = tfp.sts.Seasonal(num_seasons=20, observed_time_series=y_tensor)
        
        # Add autoregressive component for financial series
        autoregressive = tfp.sts.Autoregressive(
            order=2,
            observed_time_series=y_tensor
        )
        
        # Build model with components
        self.bsts_model = tfp.sts.Sum(
            [trend, seasonal, autoregressive], 
            observed_time_series=y_tensor
        )
        
        # Run variational inference
        self.bsts_results = tfp.sts.fit_with_hmc(self.bsts_model, y_tensor)
        
        return self

    def _predict_bayesian_sts(self, steps=1):
        """Generate BSTS forecast with uncertainty"""
        # Generate forecast distribution
        forecast_dist = tfp.sts.forecast(
            self.bsts_model,
            self.bsts_results.states_posterior_samples,
            steps,
            observed_time_series=self.bsts_model.observed_time_series
        )
        
        # Get mean prediction
        forecast_mean = forecast_dist.mean().numpy()
        
        if self.uncertainty:
            # Store uncertainty intervals
            forecast_samples = forecast_dist.sample(50).numpy()
            self.bsts_uncertainty = {
                'lower': np.percentile(forecast_samples, 5, axis=0),
                'upper': np.percentile(forecast_samples, 95, axis=0)
            }
            
        return forecast_mean[-1] if steps == 1 else forecast_mean

    def _get_oof_predictions(self, name, model, X, y, dates=None):
        """Generate out-of-fold predictions using cross-validation"""                
        from sklearn.model_selection import TimeSeriesSplit
        kf = TimeSeriesSplit(n_splits=self.n_splits)
        oof_predictions = np.zeros(len(y))
        
        # Special handling for non-standard models
        if name == 'mcmc':
            if NUMPYRO_AVAILABLE:
                # First, we need to fit the MCMC model
                self._fit_mcmc_neural_net(X, y)
                # Then we can make predictions
                mcmc_pred = self._predict_mcmc_neural_net(X)
                if isinstance(mcmc_pred, np.ndarray):
                    # Make sure the shape is correct
                    if mcmc_pred.shape[0] == len(y):
                        return mcmc_pred
                    else:
                        # Reshape if needed
                        return np.full(len(y), mcmc_pred.mean())
                else:
                    return np.full(len(y), mcmc_pred)
            else:
                return np.zeros(len(y)) 
    #3Ensemble.py
        elif name == 'garch':
            if ARCH_AVAILABLE:
                # For time series models, use simple moving window validation
                window_size = len(y) // self.n_splits
                for i in range(self.n_splits):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size
                    if end_idx > len(y) - 1:
                        end_idx = len(y) - 1
                
                    train_y = y[0:end_idx]
                    self._fit_advanced_garch(train_y)
                    oof_predictions[start_idx:end_idx] = self._predict_advanced_garch()
                
                return oof_predictions
            else:
                return np.zeros(len(y))
                
        elif name == 'prophet':
            if PROPHET_AVAILABLE:
                if dates is None:
                    dates = pd.date_range(end='today', periods=len(y))
                
                window_size = len(y) // self.n_splits
                for i in range(self.n_splits):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size
                    if end_idx > len(y) - 1:
                        end_idx = len(y) - 1
                    
                    train_y = y[0:end_idx]
                    train_dates = dates[0:end_idx]
                    
                    self._fit_prophet(train_dates, train_y)
                    oof_predictions[start_idx:end_idx] = self._predict_prophet(end_idx - start_idx)
                    
                return oof_predictions
            else:
                return np.zeros(len(y))
                
        elif name == 'bsts':
            if TFP_AVAILABLE:
                window_size = len(y) // self.n_splits
                for i in range(self.n_splits):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size
                    if end_idx > len(y) - 1:
                        end_idx = len(y) - 1
                    
                    train_y = y[0:end_idx]
                    
                    self._fit_bayesian_sts(train_y)
                    oof_predictions[start_idx:end_idx] = self._predict_bayesian_sts(end_idx - start_idx)
                    
                return oof_predictions
            else:
                return np.zeros(len(y))
        else:
            # Standard ML models
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Scale data if needed
                if name in self.models_need_scaling:
                    X_train = self.scaler.transform(X_train)
                    X_val = self.scaler.transform(X_val)
                
                # Special handling for NearestCentroid (classification algorithm being adapted)
                if name == 'nc':
                    # Convert y to discrete classes for NearestCentroid (using quantiles)
                    from sklearn.preprocessing import KBinsDiscretizer
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                    y_train_binned = discretizer.fit_transform(y_train.reshape(-1, 1)).flatten()
                    model.fit(X_train, y_train_binned)
                    
                    # Convert predictions back to original scale 
                    y_pred_binned = model.predict(X_val)
                    
                    # Calculate bin means correctly by finding indices for each bin
                    bin_means = []
                    for i in range(5):
                        bin_indices = y_train_binned == i
                        if np.any(bin_indices):
                            bin_means.append(y_train[bin_indices].mean())
                        else:
                            bin_means.append(y_train.mean())  # Fallback if a bin is empty
                    
                    oof_predictions[val_idx] = [bin_means[int(i)] for i in y_pred_binned]
                else:
                    # Standard fit/predict for other models
                    model.fit(X_train, y_train)
                    oof_predictions[val_idx] = model.predict(X_val)
            
            return oof_predictions 

  

 

#4Ensemble.py
    def fit(self, X, y, dates=None):
        """Fit the ensemble model with financial enhancements"""
        X = np.array(X)
        y = np.array(y)
       
        # Apply feature engineering if enabled
        if self.feature_engineering:
            X = self._add_financial_features(X)
       
        # Fit scaler on all data
        self.scaler.fit(X)
       
        # Generate meta-features through cross-validation
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
       
        # Generate out-of-fold predictions for meta-features
        for i, (name, model) in enumerate(self.base_models):
            meta_features[:, i] = self._get_oof_predictions(name, model, X, y, dates)
       
        # Train base models on the entire dataset
        self.trained_base_models = []
        for name, model in self.base_models:
            # Special handling for non-standard models
            if name == 'mcmc':
                if NUMPYRO_AVAILABLE:
                    self._fit_mcmc_neural_net(X, y)
                    self.trained_base_models.append((name, 'mcmc_fitted'))
            elif name == 'garch':
                if ARCH_AVAILABLE:
                    self._fit_advanced_garch(y)
                    self.trained_base_models.append((name, 'garch_fitted'))
            elif name == 'prophet':
                if PROPHET_AVAILABLE:
                    if dates is None:
                        dates = pd.date_range(end='today', periods=len(y))
                    self._fit_prophet(dates, y)
                    self.trained_base_models.append((name, 'prophet_fitted'))
            elif name == 'bsts':
                if TFP_AVAILABLE:
                    self._fit_bayesian_sts(y)
                    self.trained_base_models.append((name, 'bsts_fitted'))
            elif name == 'nc':
                # Handle NearestCentroid specially
                from sklearn.preprocessing import KBinsDiscretizer
                discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten()
               
                if name in self.models_need_scaling:
                    model_clone = NearestCentroid(metric='manhattan')
                    model_clone.fit(self.scaler.transform(X), y_binned)
                else:
                    model_clone = NearestCentroid(metric='manhattan')
                    model_clone.fit(X, y_binned)
               
                self.trained_base_models.append((name, model_clone))
                # Store discretizer for later use
                self.nc_discretizer = discretizer
               
                # Calculate bin means correctly
                self.nc_bin_means = []
                y_binned = discretizer.transform(y.reshape(-1, 1)).flatten()
                for i in range(5):
                    bin_indices = y_binned == i
                    if np.any(bin_indices):
                        self.nc_bin_means.append(y[bin_indices].mean())
                    else:
                        self.nc_bin_means.append(y.mean())  # Fallback if a bin is empty
            elif name == 'gpr':
                # Special handling for GaussianProcessRegressor with compound kernel
                if name in self.models_need_scaling:
                    # Create a new GPR with financial-specific kernel
                    kernel = 1.0 * RBF(length_scale=1.0) + 1.0 * DotProduct() + WhiteKernel(noise_level=0.1)
                    model_clone = GaussianProcessRegressor(
                        kernel=kernel, 
                        random_state=self.random_state,
                        normalize_y=True
                    )
                    model_clone.fit(self.scaler.transform(X), y)
                else:
                    kernel = 1.0 * RBF(length_scale=1.0) + 1.0 * DotProduct() + WhiteKernel(noise_level=0.1)
                    model_clone = GaussianProcessRegressor(
                        kernel=kernel, 
                        random_state=self.random_state,
                        normalize_y=True
                    )
                    model_clone.fit(X, y)
                self.trained_base_models.append((name, model_clone))
            else:
                # Standard models
                if name in self.models_need_scaling:
                    try:
                        model_clone = model.__class__(**model.get_params())
                        model_clone.fit(self.scaler.transform(X), y)
                    except TypeError:
                        # Fallback for models with complex parameters
                        model_clone = model.__class__()  # Create with defaults
                        model_clone.fit(self.scaler.transform(X), y)
                else:
                    try:
                        model_clone = model.__class__(**model.get_params())
                        model_clone.fit(X, y)
                    except TypeError:
                        # Fallback for models with complex parameters
                        model_clone = model.__class__()  # Create with defaults
                        model_clone.fit(X, y)
                self.trained_base_models.append((name, model_clone))
       
        # Train meta-model on meta-features
        self.meta_model.fit(meta_features, y)
       
        # Store meta-feature importances if supported by meta-model
        if hasattr(self.meta_model, 'feature_importances_'):
            self.meta_feature_importances = dict(zip(
                [name for name, _ in self.base_models],
                self.meta_model.feature_importances_
            ))
           
        return self
   
    def predict(self, X, return_uncertainty=False):
        """Predict with uncertainty estimation"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Apply feature engineering if enabled
        if self.feature_engineering:
            X = self._add_financial_features(X, is_training=False)
        
        # Generate meta-features from base models
        meta_features = np.zeros((n_samples, len(self.trained_base_models)))
        model_predictions = []  # Store individual model predictions for uncertainty
        
        for i, (name, model) in enumerate(self.trained_base_models):
            if name == 'mcmc':
                if NUMPYRO_AVAILABLE:
                    pred = self._predict_mcmc_neural_net(X)
                    meta_features[:, i] = pred
                    model_predictions.append(pred)
            elif name == 'garch':
                if ARCH_AVAILABLE:
                    # For time series models, just use the forecast
                    pred = self._predict_advanced_garch()
                    meta_features[:, i] = pred
                    model_predictions.append(np.full(n_samples, pred))
            elif name == 'prophet':
                if PROPHET_AVAILABLE:
                    pred = self._predict_prophet(n_samples)
                    if n_samples == 1:
                        pred = np.array([pred])
                    meta_features[:, i] = pred
                    model_predictions.append(pred)
            elif name == 'bsts':
                if TFP_AVAILABLE:
                    pred = self._predict_bayesian_sts(n_samples)
                    if n_samples == 1:
                        pred = np.array([pred])
                    meta_features[:, i] = pred
                    model_predictions.append(pred)
            elif name == 'nc':
                # Handle NearestCentroid specially
                if name in self.models_need_scaling:
                    X_scaled = self.scaler.transform(X)
                    y_pred_binned = model.predict(X_scaled)
                else:
                    y_pred_binned = model.predict(X)
                
                # Map bin predictions to original scale using stored bin means
                pred = np.array([self.nc_bin_means[int(bin_idx)] for bin_idx in y_pred_binned])
                meta_features[:, i] = pred
                model_predictions.append(pred)
            else:
                # Standard models
                if name in self.models_need_scaling:
                    pred = model.predict(self.scaler.transform(X))
                    meta_features[:, i] = pred
                    model_predictions.append(pred)
                else:
                    pred = model.predict(X)
                    meta_features[:, i] = pred
                    model_predictions.append(pred)
        
        # Use meta-model to make final predictions
        final_predictions = self.meta_model.predict(meta_features)
        
        if return_uncertainty or self.uncertainty:
            # Calculate ensemble uncertainty as standard deviation of model predictions
            uncertainty = np.std(np.array(model_predictions), axis=0)
            
            # Calculate prediction intervals (assuming approximate normality)
            lower_bound = final_predictions - 1.96 * uncertainty
            upper_bound = final_predictions + 1.96 * uncertainty
            
            return final_predictions, {
                'uncertainty': uncertainty,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return final_predictions 


  

    def predict_with_contributions(self, X):
        """Predict with individual model contributions"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Apply feature engineering if enabled
        if self.feature_engineering:
            X = self._add_financial_features(X, is_training=False)
        
        # Generate meta-features from base models
        meta_features = np.zeros((n_samples, len(self.trained_base_models)))
        contributions = {}
        
        for i, (name, model) in enumerate(self.trained_base_models):
            if name == 'mcmc':
                if NUMPYRO_AVAILABLE:
                    pred = self._predict_mcmc_neural_net(X)
                    meta_features[:, i] = pred
                    contributions[name] = pred
            elif name == 'garch':
                if ARCH_AVAILABLE:
                    pred = self._predict_advanced_garch()
                    meta_features[:, i] = pred
                    contributions[name] = np.full(n_samples, pred)
            elif name == 'prophet':
                if PROPHET_AVAILABLE:
                    pred = self._predict_prophet(n_samples)
                    if n_samples == 1:
                        pred = np.array([pred])
                    meta_features[:, i] = pred
                    contributions[name] = pred
            elif name == 'bsts':
                if TFP_AVAILABLE:
                    pred = self._predict_bayesian_sts(n_samples)
                    if n_samples == 1:
                        pred = np.array([pred])
                    meta_features[:, i] = pred
                    contributions[name] = pred
            elif name == 'nc':
                # Handle NearestCentroid specially
                if name in self.models_need_scaling:
                    X_scaled = self.scaler.transform(X)
                    y_pred_binned = model.predict(X_scaled)
                else:
                    y_pred_binned = model.predict(X)
                
                # Map bin predictions to original scale using stored bin means
                pred = np.array([self.nc_bin_means[int(bin_idx)] for bin_idx in y_pred_binned])
                meta_features[:, i] = pred
                contributions[name] = pred
            else:
                # Standard models
                if name in self.models_need_scaling:
                    pred = model.predict(self.scaler.transform(X))
                    meta_features[:, i] = pred
                    contributions[name] = pred
                else:
                    pred = model.predict(X)
                    meta_features[:, i] = pred
                    contributions[name] = pred
        
        # Get final ensemble prediction
        final_prediction = self.meta_model.predict(meta_features)
        
        # Calculate feature importance weights if available
        if hasattr(self, 'meta_feature_importances'):
            importance_weighted = {}
            for name, importance in self.meta_feature_importances.items():
                if name in contributions:
                    importance_weighted[name] = {
                        'prediction': contributions[name],
                        'importance': importance,
                        'weighted_contribution': contributions[name] * importance
                    }
            contributions = importance_weighted
            
        return final_prediction, contributions
    
    def evaluate(self, X, y):
        """Evaluate model with financial metrics"""
        # Get predictions, handling the case where uncertainty is returned
        result = self.predict(X, return_uncertainty=False)
        
        # If uncertainty is still returned (because self.uncertainty=True)
        if isinstance(result, tuple):
            predictions = result[0]  # Extract just the predictions
        else:
            predictions = result
        
        # Calculate standard regression metrics
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy (financial specific)
        dir_accuracy = self.directional_accuracy(y, predictions)
        
        # Calculate maximum drawdown (financial specific)
        cumulative_returns_true = np.cumsum(y)
        cumulative_returns_pred = np.cumsum(predictions)
        
        rolling_max_true = np.maximum.accumulate(cumulative_returns_true)
        drawdowns_true = (rolling_max_true - cumulative_returns_true) / rolling_max_true
        max_drawdown_true = np.max(drawdowns_true)
        
        rolling_max_pred = np.maximum.accumulate(cumulative_returns_pred)
        drawdowns_pred = (rolling_max_pred - cumulative_returns_pred) / rolling_max_pred
        max_drawdown_pred = np.max(drawdowns_pred)
        
        # Return comprehensive metrics
        return {
            'rmse': rmse,
            'directional_accuracy': dir_accuracy,
            'max_drawdown_true': max_drawdown_true,
            'max_drawdown_pred': max_drawdown_pred,
            'drawdown_error': abs(max_drawdown_true - max_drawdown_pred)
        }

def main():
   df = pd.read_csv('Nasdaq.csv')
   X = df.iloc[1:-1, 1:-1]
   y = df.iloc[1:-1, -1]
   new_X = df.iloc[-1:, 1:-1].astype(float)
   
   # Initialize and train the enhanced financial ensemble model
   print("Training Financial Ensemble Model...")
   ensemble = FinancialStackingEnsembleRegressor(
       n_splits=5, 
       random_state=42,
       meta_learner='gbm',  # Use GBM as meta-learner
       uncertainty=True,    # Calculate prediction uncertainty
       feature_engineering=True  # Enable financial feature engineering
   )
   ensemble.fit(X, y)
      
   # Individual model contributions
   print("\nIndividual model contributions:")
   _, contributions = ensemble.predict_with_contributions(new_X)
   
   for name, info in contributions.items():
       if isinstance(info, dict):  # If we have weighted contributions
           pred_value = info['prediction'][0] if hasattr(info['prediction'], '__len__') else info['prediction']
           print(f"{name}: {pred_value:.2f} (Importance: {info['importance']:.2f})")
       else:  # Simple contributions
           pred_value = info[0] if hasattr(info, '__len__') else info
           print(f"{name}: {pred_value:.2f}")
   
   # Evaluate on training data
   print("\nModel Evaluation:")
   metrics = ensemble.evaluate(X, y)
   for metric_name, value in metrics.items():
       print(f"{metric_name}: {value:.2f}")
           
   # Make prediction with uncertainty
   print("Making predictions with uncertainty...")
   y_new_pred, uncertainty = ensemble.predict(new_X, return_uncertainty=True)
   print(f"Predicted value: {y_new_pred[0]:.2f}")
   print(f"Uncertainty range: {uncertainty['lower_bound'][0]:.2f} to {uncertainty['upper_bound'][0]:.2f}")

if __name__ == "__main__":
   main() 
