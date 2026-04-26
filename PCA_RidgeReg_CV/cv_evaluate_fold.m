function rmse = cv_evaluate_fold(val_data, modelParameters)
% cv_evaluate_fold  Evaluate a trained model on a validation fold.
%
%   rmse = cv_evaluate_fold(val_data, modelParameters)
%
%   Mimics testFunction_for_students_MTb: decodes hand position iteratively
%   from t=320 to end of trial in 20 ms steps and computes RMSE (mm)
%   averaged across all timepoints, trials, and directions.
%
%   Inputs
%   ------
%   val_data        : (nVal x nDir) struct with .spikes, .handPos, .trialId
%   modelParameters : output of cv_train_model
%
%   Output
%   ------
%   rmse : scalar RMSE in mm

    binSize = modelParameters.binSize;
    tStart  = modelParameters.tStart;

    [nVal, nDir] = size(val_data);

    sq_err_sum = 0;
    n_points   = 0;

    for d = 1:nDir
        for tr = 1:nVal
            sp = val_data(tr,d).spikes;
            hp = val_data(tr,d).handPos;
            T  = size(sp, 2);

            if T < tStart
                continue;
            end

            % Reset persistent state before each trial
            clear positionEstimator; %#ok<CLFUNC>

            times = tStart : binSize : T;

            for ti = 1:numel(times)
                t = times(ti);

                % Build test_data struct with spikes up to t
                test_data.spikes  = sp(:, 1:t);
                test_data.trialId = val_data(tr,d).trialId;

                % Call local positionEstimator (same folder)
                [x_pred, y_pred] = positionEstimator(test_data, modelParameters);

                if t <= size(hp, 2)
                    x_true = hp(1, t);
                    y_true = hp(2, t);
                    sq_err_sum = sq_err_sum + (x_pred - x_true)^2 + (y_pred - y_true)^2;
                    n_points   = n_points + 1;
                end
            end
        end
    end

    if n_points == 0
        rmse = NaN;
    else
        rmse = sqrt(sq_err_sum / n_points);
    end

end
