x = [9 2.140641477955609996536345; ...
	2.5 0.7031566406452431872257; ...
	0.1 -10.42375494041107679516822; ...
	7e-4 -1429.147493371120205005198; ...
	7e-5 -14286.29138623969227538398; ...
	7e-6 -142857.7200612932791081972; ...
	2e-6 -500000.5772123750382073831; ...
	1e-6 -1000000.577214019968668068; ...
	7e-7 -1428572.005785942019703646; ...
	-0.5 .03648997397857652055902367; ...
	-1.1 10.15416395914385769902271 ...
	];
for i = 1:rows(x)
	actual = digamma(x(i,1));
	expected = x(i,2);
	e = abs(actual - expected)/expected;
	if e > 1e-12
		error(sprintf('digamma(%g) = %g should be %g', x(i,1), actual, expected));
	end
end
if digamma(-1) ~= -Inf
	error('digamma(-1) should be -Inf');
end
if digamma(0) ~= -Inf
  error('digamma(0) should be -Inf');
end
if ~isnan(digamma(-Inf))
  error('digamma(-Inf) should be NaN');
end
