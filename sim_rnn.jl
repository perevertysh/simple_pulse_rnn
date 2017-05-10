#import PyPlot

type impulse_data
  x::Array{Float32, 2}
  y::Array{Float32, 1}

  function impulse_data(data_len::Int64)
    x = zeros(Float32, data_len, 2)
    y = zeros(Float32, data_len)

    period::Int64 = 11
    i::Int64 = 1
    pulse_len::Int64 = 0

    while i <= data_len
      a = rand(0:1)
      x[i, 1] = a
      if a == 0
        b = rand()
        x[i, 2] = b
        pulse_len = round(b * 10)
        if i + pulse_len <= data_len
          y[i:(i + pulse_len)] = 1
        else
          y[i:end] = 1
        end
        i += period
      end
      i += 1
    end

    new(x, y)
  end

end

function create_data(data_len::Int64)
  x = zeros(data_len, 2)
  y = zeros(data_len)
  period::Int64 = 11
  i::Int64 = 1
  pulse_len::Int64 = 0
  while i <= data_len
    a = rand(0:1)
    x[i, 1] = a
    if a == 0
      b = rand()
      x[i, 2] = b
      pulse_len = round(b * 10)
      if i + pulse_len <= data_len
        y[i:(i + pulse_len)] = 1
      else
        y[i:end] = 1
      end
      i += period
    end
    i += 1
  end
  return x, y
end

x, y = create_data(100000000)
#PyPlot.plot(x[:, 1])
#PyPlot.plot(y)
#PyPlot.show()
