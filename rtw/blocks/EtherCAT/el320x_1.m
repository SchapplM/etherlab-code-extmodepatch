classdef el320x_1 < EtherCATSlave


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        %====================================================================
        function obj = el320x_1(id)
            if nargin > 0
                obj.slave = obj.find(id);
            end
        end

        %====================================================================
        function rv = configure(obj,status,vector,filter,...
                                element,technology,resistance)

            rv.SlaveConfig.vendor = 2;
            rv.SlaveConfig.description = obj.slave{1};
            rv.SlaveConfig.product  = obj.slave{2};

            pdo_count = str2double(obj.slave{1}(6));

            pdo = el320x_1.pdo;
            rv.SlaveConfig.sm = {{3,1, arrayfun(@(i) pdo(i,:), ...
                                         obj.slave{5} + (0:pdo_count-1),...
                                         'UniformOutput', false)}};

            idx = size(pdo{obj.slave{5},2},1);
            pdo = repmat([0,0,idx-1,0],pdo_count,1);
            pdo(:,2) = 0:pdo_count-1;

            rv.PortConfig.output = ...
                el320x_1.configurePorts('Ch.',pdo,sint(16),vector);

            port_count = numel(rv.PortConfig.output);

            if status
                pdo(:,3) = obj.slave{6};

                if vector
                    n = 1;
                else
                    n = pdo_count;
                end
                
                rv.PortConfig.output(end+(1:n)) = el320x_1.configurePorts(...
                        'St.',pdo,uint(1),vector);
            end

            % Set the full scale range for temperature port
            for i = 1:port_count
                rv.PortConfig.output(i).full_scale = obj.slave{4};
            end
            
            subIndex = hex2dec({'19','1a','1b'})';
            config = [hex2dec('8000') + repmat(0:pdo_count-1,1,3)*16;
                      subIndex(floor((0:3*pdo_count-1)/pdo_count)+1);
                      repmat(16,1,pdo_count*3);
                      [element(1:pdo_count), technology(1:pdo_count),...
                       resistance(1:pdo_count)]]';

            rv.SlaveConfig.sdo = ...
                num2cell( [hex2dec('8000'),hex2dec(' 6'), 8,double(filter > 1);
                           hex2dec('8000'),hex2dec('15'),16,max(0,filter-2);
                           config]);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Static)
        %====================================================================
        function modelChanged()
            obj = el320x_1(get_param(gcbh,'model'));
            pdo_count = str2double(obj.slave{1}(6));
            obj.updateSDOEnable(dec2base(1:3*pdo_count,10,2));
            obj.updateRevision();
        end

        %====================================================================
        function test(p)
            ei = EtherCATInfo(fullfile(p,'Beckhoff EL32xx.xml'));
            for i = 1:size(el320x_1.models,1)
                fprintf('Testing %s\n', el320x_1.models{i,1});
                slave = ei.getSlave(el320x_1.models{i,2},...
                        'revision', el320x_1.models{i,3});
                model = el320x_1.models{i,1};

                rv = el320x_1(model).configure(i&1,i&2,...
                        1,1:8,1:8,1:8);
                slave.testConfig(rv.SlaveConfig,rv.PortConfig);
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties (Constant)
        %              PdoEntry       EntryIdx, SubIdx, Bitlen
        pdo = { hex2dec('1a00'), [hex2dec('6000'), 1, 1;
                                  hex2dec('6000'), 2, 1;
                                  hex2dec('6000'), 3, 2;
                                  hex2dec('6000'), 5, 2;
                                  hex2dec('6000'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('1800'), 7, 1;
                                  hex2dec('1800'), 9, 1;
                                  hex2dec('6000'),17,16];
                hex2dec('1a01'), [hex2dec('6010'), 1, 1;
                                  hex2dec('6010'), 2, 1;
                                  hex2dec('6010'), 3, 2;
                                  hex2dec('6010'), 5, 2;
                                  hex2dec('6010'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('1801'), 7, 1;
                                  hex2dec('1801'), 9, 1;
                                  hex2dec('6010'),17,16];
                hex2dec('1a02'), [hex2dec('6020'), 1, 1;
                                  hex2dec('6020'), 2, 1;
                                  hex2dec('6020'), 3, 2;
                                  hex2dec('6020'), 5, 2;
                                  hex2dec('6020'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('1802'), 7, 1;
                                  hex2dec('1802'), 9, 1;
                                  hex2dec('6020'),17,16];
                hex2dec('1a03'), [hex2dec('6030'), 1, 1;
                                  hex2dec('6030'), 2, 1;
                                  hex2dec('6030'), 3, 2;
                                  hex2dec('6030'), 5, 2;
                                  hex2dec('6030'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('1803'), 7, 1;
                                  hex2dec('1803'), 9, 1;
                                  hex2dec('6030'),17,16];
                hex2dec('1a00'), [hex2dec('6000'), 1, 1;
                                  hex2dec('6000'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6000'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6000'),15, 1;
                                  hex2dec('6000'),16, 1;
                                  hex2dec('6000'),17,16];
                hex2dec('1a01'), [hex2dec('6010'), 1, 1;
                                  hex2dec('6010'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6010'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6010'),15, 1;
                                  hex2dec('6010'),16, 1;
                                  hex2dec('6010'),17,16];
                hex2dec('1a02'), [hex2dec('6020'), 1, 1;
                                  hex2dec('6020'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6020'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6020'),15, 1;
                                  hex2dec('6020'),16, 1;
                                  hex2dec('6020'),17,16];
                hex2dec('1a03'), [hex2dec('6030'), 1, 1;
                                  hex2dec('6030'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6030'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6030'),15, 1;
                                  hex2dec('6030'),16, 1;
                                  hex2dec('6030'),17,16];
                hex2dec('1a04'), [hex2dec('6040'), 1, 1;
                                  hex2dec('6040'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6040'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6040'),15, 1;
                                  hex2dec('6040'),16, 1;
                                  hex2dec('6040'),17,16];
                hex2dec('1a05'), [hex2dec('6050'), 1, 1;
                                  hex2dec('6050'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6050'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6050'),15, 1;
                                  hex2dec('6050'),16, 1;
                                  hex2dec('6050'),17,16];
                hex2dec('1a06'), [hex2dec('6060'), 1, 1;
                                  hex2dec('6060'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6060'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6060'),15, 1;
                                  hex2dec('6060'),16, 1;
                                  hex2dec('6060'),17,16];
                hex2dec('1a07'), [hex2dec('6070'), 1, 1;
                                  hex2dec('6070'), 2, 1;
                                  0              , 0, 4;
                                  hex2dec('6070'), 7, 1;
                                  0              , 0, 7;
                                  hex2dec('6070'),15, 1;
                                  hex2dec('6070'),16, 1;
                                  hex2dec('6070'),17,16];
        };

        %   Model   ProductCode           RevisionNo          TempScale, PDOIdx, Stat.Idx
        models = {
          'EL3201',       hex2dec('0c813052'), hex2dec('00100000'),  10,      1, 4;
          'EL3201-0010',  hex2dec('0c813052'), hex2dec('0010000a'), 100,      1, 4;
          'EL3201-0020',  hex2dec('0c813052'), hex2dec('00100014'), 100,      1, 4;
          'EL3202',       hex2dec('0c823052'), hex2dec('00100000'),  10,      1, 4;
          'EL3202-0010',  hex2dec('0c823052'), hex2dec('0010000a'), 100,      1, 4;
          'EL3202-0020',  hex2dec('0c823052'), hex2dec('00150014'),  10,      1, 4;
          'EL3204',       hex2dec('0c843052'), hex2dec('00100000'),  10,      1, 4;
          'EL3204-0200',  hex2dec('0c843052'), hex2dec('001100c8'),  10,      5, 3;
          'EL3208',       hex2dec('0c883052'), hex2dec('00110000'),  10,      5, 3;
          'EL3214',       hex2dec('0c8e3052'), hex2dec('00100000'),  10,      5, 3;
        };
    end
end
