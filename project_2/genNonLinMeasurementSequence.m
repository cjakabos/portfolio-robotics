function Y = genNonLinearMeasurementSequence(X, h, R)
    % GENNONLINEARMEASUREMENTSEQUENCE generates ovservations of the states 
    % sequence X using a non-linear measurement model.
    %
    % Input:
    %   X           [n x N+1] State vector sequence
    %   h           Measurement model function handle
    %   h           Measurement model function handle
    %               [hx,Hx]=h(x) 
    %               Takes as input x (state) 
    %               Returns hx and Hx, measurement model and Jacobian evaluated at x
    %   R           [m x m] Measurement noise covariance
    %
    % Output:
    %   Y           [m x N] Measurement sequence

    m = size(R,1);
    % state sequence
    N = size(X,2);
    Y = zeros(m,N);
    
    % iterate to generate N-1 samples (exclude x0)
    for i=1:N-1
        % Measurement model
        Y(:,i) = h(X(:,i)) + mvnrnd(zeros(m,1), R)';
    end
end