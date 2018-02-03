function [tInd,eInd] = crossvalind_2(method,N,varargin)

classesProvided = false;
MG = 1;
P = 0.5;
K = 5;
M = 1;
Q = [1 1];

if ischar(method) && size(method,1)==1
    validMethods = {'holdout','kfold','resubstitution','leavemout'};
    method = strmatch(lower(method),validMethods);
    if isempty(method)
        error('Bioinfo:crossvalind:NotValidMethod',...
            'Not a valid method.')
    end
    method = validMethods{method};
else
    error('Bioinfo:crossvalind:NotValidTypeForMethod',...
        'Valid methods are ''KFold'', ''HoldOut'', ''LeaveMOut'', or ''Resubstitution''.')
end

if nargout>1 && isequal(method,'kfold')
    error('Bioinfo:crossvalind:TooManyOutputArgumentsForKfold',...
        'To many output arguments for Kfold cross-validation.')
end

if numel(varargin) && isnumeric(varargin{1})
    S = varargin{1};
    varargin(1)=[];
    switch method
        case 'holdout'
            if numel(S)==1 && S>0 && S<1
                P = S;
            else
                error('Bioinfo:crossvalind:InvalidThirdInputP',...
                    'For hold-out cross-validation, the third input must be a scalar between 0 and 1.');
            end
        case 'kfold'
            if  numel(S)==1 && S>=1
                K = round(S);
            else
                error('Bioinfo:crossvalind:InvalidThirdInputK',...
                    'For Kfold cross-validation, the third input must be a positive integer.');
            end
        case 'leavemout'
            if  numel(S)==1 && S>=1
                M = round(S);
            else
                error('Bioinfo:crossvalind:InvalidThirdInputM',...
                    'For leave-M-out cross-validation, the third input must be a positive integer.');
            end
        case 'resubstitution'
            if numel(S)==2 && all(S>0) && all(S<=1)
                Q = S(:);
            else
                error('Bioinfo:crossvalind:InvalidThirdInputQ',...
                    'For resubstitution cross-validation, the third input must be a 2x1 vector with values between 0 and 1.');
            end
    end
end

if numel(varargin)
    if rem(numel(varargin),2)
        error('Bioinfo:crossvalind:IncorrectNumberOfArguments',...
            'Incorrect number of arguments to %s.',mfilename);
    end
    okargs = {'classes','min'};
    for j=1:2:numel(varargin)
        pname = varargin{j};
        pval = varargin{j+1};
        k = strmatch(lower(pname), okargs);
        if isempty(k)
            error('Bioinfo:crossvalind:UnknownParameterName',...
                'Unknown parameter name: %s.',pname);
        elseif length(k)>1
            error('Bioinfo:crossvalind:AmbiguousParameterName',...
                'Ambiguous parameter name: %s.',pname);
        else
            switch(k)
                case 1
                    classesProvided = true;
                    classes = pval;
                case 2
                    MG = round(pval(1));
                    if MG<0
                        error('Bioinfo:crossvalind:NotValidMIN',...
                            'MIN must be a positive scalar.')
                    end
            end
        end
    end
end

if isscalar(N) && isnumeric(N)
    if N<1 || N~=floor(N)
        error('Bioinfo:crossvalind:NNotPositiveInteger',...
            'The number of observations must be a positive integer.');
    end
    group = ones(N,1);
else
    [group, groupNames] = grp2idx(N);
    N = numel(group);
end

if classesProvided
    orgN = N;
    [dummy,classes]=grp2idx(classes);
    validGroups = intersect(classes,groupNames);
    if isempty(validGroups)
        error('bioinfo:crossvalind:EmptyValidGroups',...
            'Could not find any valid group. Are CLASSES the same type as GROUP ?')
    end
    selectedGroups = ismember(groupNames(group),validGroups);
    group = grp2idx(group(selectedGroups));
    N = numel(group);
end

nS = accumarray(group(:),1);
if min(nS)<MG
    error('Bioinfo:crossvalind:MissingObservations',...
        'All the groups must have at least least MIN obeservation(s).')
end

switch method
    case {'leavemout','holdout','resubstitution'}
        switch method
            case 'leavemout'
                nSE = repmat(M,numel(nS),1);
                nST = max(nS-nSE,MG);
            case 'holdout'
                nSE = floor(nS*P);
                nST = max(nS-nSE,MG);
            case 'resubstitution'
                nSE = floor(nS*Q(1));
                nST = floor(nS*Q(2));
                nST = max(nST,MG);
        end
        tInd = false(N,1);
        eInd = false(N,1);
        for g = 1:numel(nS)
            h = find(group==g);
            randInd = randperm(nS(g));
            tInd(h(randInd(1:nST(g))))=true;
            eInd(h(randInd(end-nSE(g)+1:end)))=true;
        end
    case 'kfold'
        tInd = zeros(N,1);
        for g = 1:numel(nS)
            h = find(group==g);
            q = ceil(K*(1:nS(g))/nS(g));
            pq = randperm(K);
            randInd = randperm(nS(g));
            tInd(h(randInd))=pq(q);
        end
end

if classesProvided
    if isequal(method,'kfold')
        temp = zeros(orgN,1);
        temp(selectedGroups) = tInd;
        tInd = temp;
    else
        temp = false(orgN,1);
        temp(selectedGroups) = tInd;
        tInd = temp;
        temp = false(orgN,1);
        temp(selectedGroups) = eInd;
        eInd = temp;
    end
end